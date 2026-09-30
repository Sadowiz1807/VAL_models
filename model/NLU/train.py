"""NLU annotation, masked losses, evaluation, calibration, expansion and train CLI.

Nothing trains on import. Manifests are JSONL; see docs/NLUModel_architecture.md.
Each row also needs an explicit boolean `ood` label. Optional `group_id` keeps
paraphrase/augmentation families in a single split. Overflow rows keep ALL gold
operations, supervise count=K+1 and mask operation extraction (never truncate).
Character tokenizer is fitted only on training text. Calibration is a separate
command and requires a held-out manifest, not training/validation data.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

if __package__:
    from .config import (create_config, load_config, save_config, validate_nlu_config,
                         build_registry, fingerprint, SOURCES)
    from .model import (CharacterTokenizer, build_model, load_for_inference, normalize_text,
                        validate_parameter_value, parse_number, decode_prediction, decode_goal_action, validate_calibration)
else:
    from config import (create_config, load_config, save_config, validate_nlu_config,
                        build_registry, fingerprint, SOURCES)
    from model import (CharacterTokenizer, build_model, load_for_inference, normalize_text,
                       validate_parameter_value, parse_number, decode_prediction, decode_goal_action, validate_calibration)

IGNORE = -100


def read_manifest(path):
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("Empty NLU manifest")
    seen = set()
    for row in rows:
        for field in ("dataset_id", "sample_id", "text", "act", "turn_relation"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing/invalid {field}")
        if row["text"] != normalize_text(row["text"]):
            raise ValueError(f"Store NFC text and update annotation offsets first: {row['sample_id']}")
        if type(row.get("ood")) is not bool or type(row.get("context_required")) is not bool:
            raise ValueError("Explicit boolean ood/context_required annotations are required")
        if not isinstance(row.get("operations"), list):
            raise ValueError("operations must be a list")
        key = (row["dataset_id"], row["sample_id"])
        if key in seen:
            raise ValueError(f"Duplicate sample: {key}")
        seen.add(key)
    return rows


def split_identity(rows):
    return {
        "samples": sorted({fingerprint([r["dataset_id"], r["sample_id"]]) for r in rows}),
        "texts": sorted({fingerprint(normalize_text(r["text"])) for r in rows}),
        "groups": sorted({fingerprint([r["dataset_id"], r["group_id"]]) for r in rows if "group_id" in r}),
    }


def assert_disjoint(left, right):
    for kind in ("samples", "texts", "groups"):
        if set(left.get(kind, [])) & set(right.get(kind, [])):
            raise ValueError(f"Split leakage detected in {kind}")


def _index(labels, label, name):
    try:
        return labels.index(label)
    except ValueError as error:
        raise ValueError(f"Unknown {name}: {label}") from error


def _canonical(operation):
    action, goal = operation["action"], operation["goal"]
    canonical = action if action.startswith(goal + ".") else f"{goal}.{action}"
    if operation.get("canonical_action", canonical) != canonical:
        raise ValueError("Conflicting canonical_action")
    return canonical


class NLUDataset(Dataset):
    def __init__(self, rows_or_path, config, registry, tokenizer):
        self.rows = read_manifest(rows_or_path) if isinstance(rows_or_path, (str, Path)) else copy.deepcopy(rows_or_path)
        if not self.rows:
            raise ValueError("Empty NLU dataset")
        self.config, self.registry, self.tokenizer = config, registry, tokenizer
        # Eager validation catches bad spans/references before any optimization.
        self.items = [self._encode(row) for row in self.rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def _encode(self, row):
        r, k = self.registry, self.config["operation_decoder"]["max_operations"]
        s = len(r["slots"])
        if row["text"] != normalize_text(row["text"]):
            raise ValueError("Annotation offsets require stored NFC text")
        if type(row.get("ood")) is not bool or type(row.get("context_required")) is not bool:
            raise ValueError("Explicit ood/context_required labels required")
        encoded = self.tokenizer.encode(row["text"])
        ops = row["operations"]
        overflow = len(ops) > k
        target = {
            "act_y": torch.tensor(_index(r["acts"], row["act"], "act")),
            "turn_relation_y": torch.tensor(_index(r["turn_relations"], row["turn_relation"], "turn relation")),
            "context_required_y": torch.tensor(float(row["context_required"])),
            "context_reference_y": torch.tensor(IGNORE), "ood_y": torch.tensor(float(row["ood"])),
            "count_y": torch.tensor(min(len(ops), k + 1)),
            "operation_mask": torch.full((k,), not overflow, dtype=torch.bool),
            "operation_presence_y": torch.zeros(k),
            "goal_y": torch.full((k,), IGNORE, dtype=torch.long),
            "action_y": torch.full((k,), IGNORE, dtype=torch.long),
            "slot_presence_y": torch.zeros(k, s),
            "source_y": torch.full((k, s), IGNORE, dtype=torch.long),
            "span_start_y": torch.full((k, s), IGNORE, dtype=torch.long),
            "span_end_y": torch.full((k, s), IGNORE, dtype=torch.long),
            "state_path_y": torch.full((k, s), IGNORE, dtype=torch.long),
            "slot_context_y": torch.full((k, s), IGNORE, dtype=torch.long),
            "enum_y": torch.full((k, s), IGNORE, dtype=torch.long),
            "boolean_y": torch.full((k, s), float(IGNORE)),
            "operation_relation_y": torch.full((k, k), IGNORE, dtype=torch.long),
        }
        if row.get("context_reference") is not None:
            if not row["context_required"]:
                raise ValueError("Context reference supplied while context_required=False")
            target["context_reference_y"] = torch.tensor(_index(r["context_types"], row["context_reference"], "turn reference"))
        if row["act"] == "EXECUTE" and not ops:
            raise ValueError("EXECUTE annotation needs at least one interpretable operation")
        if overflow:
            # Count supervision needs true overflow examples; extraction labels cannot
            # represent them. Keep full row for evaluation, not a first-K target.
            return {"encoded": encoded, "targets": target, "row": row}
        for i, op in enumerate(ops):
            canonical = _canonical(op)
            action_id = _index(r["actions"], canonical, "action")
            goal_id = _index(r["domains"], op["goal"], "goal")
            if r["action_to_domain"][action_id] != goal_id:
                raise ValueError("Action/domain mismatch")
            target["operation_presence_y"][i] = 1
            target["goal_y"][i], target["action_y"][i] = goal_id, action_id
            parameters = op.get("parameters", {})
            slots = {slot["name"]: (j, slot["schema"]) for j, slot in enumerate(r["slots"]) if slot["action_id"] == action_id}
            if not set(parameters) <= set(slots):
                raise ValueError(f"Unknown parameter in {row['sample_id']}")
            for name, value in parameters.items():
                j, schema = slots[name]
                source = value["source"]
                source_id = _index(SOURCES, source, "source")
                target["slot_presence_y"][i, j] = 1
                target["source_y"][i, j] = source_id
                if source == "INPUT_SPAN":
                    start, end = value["char_start"], value["char_end"]
                    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(row["text"]):
                        raise ValueError(f"Invalid Unicode offsets: {row['sample_id']}:{name}")
                    if end - start > self.config["decoding"]["max_span_length"]:
                        raise ValueError("Annotated span exceeds decoder max_span_length")
                    raw = row["text"][start:end]
                    parsed = parse_number(raw) if schema["type"] == "NUMBER" else raw
                    if schema["type"] in ("ENTITY", "FREE_TEXT", "NUMBER") and value["value"] != parsed:
                        raise ValueError(f"Span/value mismatch: {row['sample_id']}:{name}")
                    validate_parameter_value(value["value"], schema, source)
                    # Character tokenizer: BOS shifts each character token by 1.
                    target["span_start_y"][i, j], target["span_end_y"][i, j] = start + 1, end
                    if schema["type"] == "ENUM":
                        target["enum_y"][i, j] = schema["values"].index(value["value"])
                    elif schema["type"] == "BOOLEAN":
                        target["boolean_y"][i, j] = float(value["value"])
                else:
                    validate_parameter_value(value["value"], schema, source)
                    if source == "STATE_REFERENCE":
                        target["state_path_y"][i, j] = _index(r["state_paths"], value["value"], "state path")
                    else:
                        target["slot_context_y"][i, j] = _index(r["context_types"], value["value"], "slot reference")
            constraints = r["action_constraints"][canonical]
            for group in constraints.get("mutually_exclusive", []):
                if sum(name in parameters for name in group) > 1:
                    raise ValueError("Conflicting annotated parameter alternatives")
            # Every eligible earlier pair is NONE unless explicitly annotated.
            target["operation_relation_y"][i, :i] = r["operation_relations"].index("NONE")
            seen_dependencies = set()
            for edge in op.get("dependencies", []):
                j = edge["operation_index"]
                if type(j) is not int or not 0 <= j < i or j in seen_dependencies:
                    raise ValueError("Dependencies must be unique edges to earlier operations")
                seen_dependencies.add(j)
                target["operation_relation_y"][i, j] = _index(r["operation_relations"], edge["relation"], "operation relation")
        return {"encoded": encoded, "targets": target, "row": row}


def collate_nlu(items):
    if not items:
        raise ValueError("Empty batch")
    ids = [torch.tensor(x["encoded"]["input_ids"], dtype=torch.long) for x in items]
    masks = [torch.tensor(x["encoded"]["span_valid_mask"], dtype=torch.bool) for x in items]
    padded = pad_sequence(ids, batch_first=True, padding_value=CharacterTokenizer.pad_id)
    return {
        "input_ids": padded, "attention_mask": padded.ne(CharacterTokenizer.pad_id),
        "span_valid_mask": pad_sequence(masks, batch_first=True, padding_value=False),
        "targets": {key: torch.stack([x["targets"][key] for x in items]) for key in items[0]["targets"]},
        "encoded": [x["encoded"] for x in items], "rows": [x["row"] for x in items],
    }


def valid_cross_entropy(logits, targets, ignore_index=IGNORE):
    logits, targets = logits.float(), targets.long()
    valid = targets.ne(ignore_index)
    if not valid.any():
        return torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits)).sum() * 0.0
    selected, labels = logits[valid], targets[valid]
    if (labels < 0).any() or (labels >= selected.size(-1)).any():
        raise ValueError("Classification target out of bounds")
    if not torch.isfinite(selected.gather(-1, labels[:, None])).all():
        raise ValueError("Gold class is masked/nonfinite; fix labels or schema masks")
    return F.cross_entropy(selected, labels)


def masked_bce(logits, targets, mask, negative_weight=1.0):
    logits, targets = logits.float(), targets.float()
    if not mask.any():
        return logits.sum() * 0
    values = F.binary_cross_entropy_with_logits(logits[mask], targets[mask], reduction="none")
    weights = torch.where(targets[mask] > 0.5, 1.0, negative_weight)
    return (values * weights).mean()


def compute_losses(model, output, targets, stage=None):
    """All components normalized over valid labels; no raw sum across 42 slots."""
    t = targets
    stage = stage or model.config["training"]["stage"]
    if stage not in model.config["stages"]:
        raise ValueError(f"Unknown stage {stage}")
    valid_ops = t["action_y"].ne(IGNORE)
    safe_goal = t["goal_y"].clamp_min(0)
    losses = {
        "act": valid_cross_entropy(output["act_logits"], t["act_y"]),
        "operation_count": valid_cross_entropy(output["count_logits"], t["count_y"]),
        "operation_presence": masked_bce(output["presence_logits"], t["operation_presence_y"], t["operation_mask"]),
        "operation_domain_retrieval": valid_cross_entropy(output["goal_logits"], t["goal_y"]),
        "action": valid_cross_entropy(model.mask_actions(output["action_logits"], safe_goal), t["action_y"]),
        "turn_relation": valid_cross_entropy(output["turn_relation_logits"], t["turn_relation_y"]),
        "context_required": F.binary_cross_entropy_with_logits(output["context_required_logit"].float(), t["context_required_y"]),
        "context_reference": valid_cross_entropy(output["context_reference_logits"], t["context_reference_y"]),
        "operation_relations": valid_cross_entropy(output["operation_relation_logits"], t["operation_relation_y"]),
        "ood": F.binary_cross_entropy_with_logits(output["ood_logit"].float(), t["ood_y"]),
    }
    # Supervise all slots of gold action plus slots of top-scoring wrong action
    # schemas. Never mark another gold operation's slots as its positives.
    action_selected = F.one_hot(t["action_y"].clamp_min(0), len(model.registry["actions"])).bool() & valid_ops[..., None]
    negative_count = min(model.config["training"]["parameter_hard_negative_schemas"], len(model.registry["actions"]) - 1)
    if negative_count:
        scores = output["action_logits"].detach().masked_fill(action_selected, float("-inf"))
        negative = scores.topk(negative_count, dim=-1).indices
        action_selected = action_selected.clone().scatter(-1, negative, True) & valid_ops[..., None]
    slot_mask = action_selected[..., model.slot_action]
    source_logits = output["source_logits"].masked_fill(~model.source_allowed[None, None], float("-inf"))
    state_logits = output["state_path_logits"].masked_fill(~model.state_allowed[None, None], float("-inf"))
    context_logits = output["slot_context_logits"].masked_fill(~model.context_allowed[None, None], float("-inf"))
    parts = {
        "presence": masked_bce(output["slot_presence_logits"], t["slot_presence_y"], slot_mask,
                               model.config["training"]["parameter_presence_negative_weight"]),
        "source": valid_cross_entropy(source_logits, t["source_y"]),
        "start": valid_cross_entropy(output["span_start_logits"], t["span_start_y"]),
        "end": valid_cross_entropy(output["span_end_logits"], t["span_end_y"]),
        "state": valid_cross_entropy(state_logits, t["state_path_y"]),
        "context": valid_cross_entropy(context_logits, t["slot_context_y"]),
        "boolean": masked_bce(output["boolean_logits"], t["boolean_y"], t["boolean_y"].ne(IGNORE)),
    }
    enum_losses = [valid_cross_entropy(logits, t["enum_y"][..., int(key)])
                   for key, logits in output["enum_logits"].items() if t["enum_y"][..., int(key)].ne(IGNORE).any()]
    parts["enum"] = torch.stack(enum_losses).mean() if enum_losses else output["act_logits"].sum() * 0
    losses["parameters"] = sum(model.config["parameter_loss_weights"][key] * value for key, value in parts.items())
    total = sum(model.config["loss_weights"][key] * losses[key] for key in model.config["stages"][stage])
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite semantic loss; inspect annotations/masks")
    return {"loss": total, **losses, **{f"parameter_{k}": v for k, v in parts.items()}}


def _device_batch(batch, device):
    return {key: batch[key].to(device) for key in ("input_ids", "attention_mask", "span_valid_mask")}, {
        key: value.to(device) for key, value in batch["targets"].items()}


def train_one_epoch(model, loader, optimizer, scaler=None, stage=None):
    model.train()
    device = next(model.parameters()).device
    config = model.config["training"]
    if config["amp"] == "fp16" and device.type != "cuda":
        raise ValueError("FP16 requires CUDA")
    scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=False)
    optimizer.zero_grad(set_to_none=True)
    totals, samples = {}, 0
    accumulation = config["gradient_accumulation_steps"]
    for index, batch in enumerate(loader):
        inputs, targets = _device_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=config["amp"] != "off",
                            dtype=torch.float16 if config["amp"] == "fp16" else torch.bfloat16):
            output = model(**inputs)
        losses = compute_losses(model, output, targets, stage)
        group_start = index // accumulation * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        scaler.scale(losses["loss"] / group_size).backward()
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        size = len(batch["rows"])
        samples += size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value.detach().item() * size
    return {key: value / max(samples, 1) for key, value in totals.items()}


def _operation_signature(operation, gold=False):
    parameters = tuple(sorted((name, p["source"], json.dumps(p["value"], ensure_ascii=False, sort_keys=True))
                              for name, p in operation.get("parameters", {}).items()))
    dependencies = tuple(sorted((edge["operation_index"], edge["relation"]) for edge in operation.get("dependencies", [])))
    return (operation["goal"], _canonical(operation) if gold else operation["canonical_action"], parameters, dependencies)


def _frame_signature(row, gold=False):
    return (row["act"], row["turn_relation"], row["context_required"], row.get("context_reference"),
            tuple((i if gold else op["operation_index"], _operation_signature(op, gold))
                  for i, op in enumerate(row["operations"])))


def calibration_metrics(probabilities, targets, bins=10):
    probabilities, targets = probabilities.float(), targets.long()
    if not len(targets):
        return {"ece": None, "brier": None, "reliability_bins": []}
    scores, predictions = probabilities.max(-1)
    correct = predictions.eq(targets).float()
    ece, rows = 0.0, []
    for i in range(bins):
        mask = (scores >= i / bins) & ((scores < (i + 1) / bins) if i + 1 < bins else (scores <= 1))
        count = int(mask.sum())
        if count:
            confidence, accuracy = float(scores[mask].mean()), float(correct[mask].mean())
            ece += count / len(targets) * abs(confidence - accuracy)
            rows.append({"lower": i / bins, "upper": (i + 1) / bins, "count": count,
                         "confidence": confidence, "accuracy": accuracy})
    gold = F.one_hot(targets, probabilities.size(-1)).float()
    order = scores.argsort(descending=True)
    curve = []
    for coverage in (0.25, 0.5, 0.75, 1.0):
        count = max(1, math.ceil(len(targets) * coverage))
        curve.append({"coverage": count / len(targets), "error": 1 - float(correct[order[:count]].mean())})
    return {"ece": ece, "brier": float(((probabilities - gold) ** 2).sum(-1).mean()),
            "reliability_bins": rows, "risk_coverage": curve}


@torch.inference_mode()
def validate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    r = model.registry
    confusion = torch.zeros(len(r["acts"]), len(r["acts"]), dtype=torch.long)
    totals, probabilities, act_targets = {}, [], []
    samples = full = accepted = accepted_correct = goal_correct = action_correct = gold_ops = 0
    count_correct = overflow_total = overflow_correct = 0
    parameter_tp = parameter_predicted = parameter_gold = 0
    relation_tp = relation_predicted = relation_gold = 0
    unknown_total = unknown_accepted = known_total = known_rejected = 0
    oracle_actions = oracle_count = 0
    for batch in loader:
        inputs, targets = _device_batch(batch, device)
        output = model(**inputs)
        losses = compute_losses(model, output, targets)
        size = len(batch["rows"])
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
        temp = model.calibration.get("temperatures", {}).get("act", 1.0)
        probabilities.append((output["act_logits"].float() / temp).softmax(-1).cpu())
        act_targets.append(targets["act_y"].cpu())
        valid = targets["action_y"].ne(IGNORE)
        oracle_prediction = model.mask_actions(output["action_logits"], targets["goal_y"].clamp_min(0)).argmax(-1)
        oracle_actions += int((oracle_prediction[valid] == targets["action_y"][valid]).sum())
        oracle_count += int(valid.sum())
        for b, (row, encoded) in enumerate(zip(batch["rows"], batch["encoded"])):
            prediction = decode_prediction(model, output, encoded, b)
            samples += 1
            exact = _frame_signature(prediction) == _frame_signature(row, gold=True)
            full += exact
            accept = prediction["status"] == "ok"
            accepted += accept
            accepted_correct += accept and exact
            confusion[r["acts"].index(row["act"]), r["acts"].index(prediction["act"])] += 1
            predicted_count = prediction["predicted_operation_count"]
            overflow = len(row["operations"]) > model.k
            count_correct += predicted_count == ("overflow" if overflow else len(row["operations"]))
            overflow_total += overflow
            overflow_correct += overflow and predicted_count == "overflow"
            if row["ood"]:
                unknown_total += 1
                unknown_accepted += prediction["ood_score"] <= model.config["thresholds"]["max_ood_score"]
            else:
                known_total += 1
                known_rejected += prediction["ood_score"] > model.config["thresholds"]["max_ood_score"]
            by_index = {op["operation_index"]: op for op in prediction["operations"]}
            predicted_parameters, gold_parameters, predicted_relations, gold_relations = set(), set(), set(), set()
            for i, op in enumerate(row["operations"]):
                gold_ops += 1
                candidate = by_index.get(i)
                goal_correct += candidate is not None and candidate["goal"] == op["goal"]
                action_correct += candidate is not None and candidate["canonical_action"] == _canonical(op)
                for name, p in op.get("parameters", {}).items():
                    gold_parameters.add((i, _canonical(op), name, p["source"], json.dumps(p["value"], sort_keys=True)))
                for edge in op.get("dependencies", []):
                    if edge["relation"] != "NONE":
                        gold_relations.add((i, edge["operation_index"], edge["relation"]))
            for op in prediction["operations"]:
                i = op["operation_index"]
                for name, p in op["parameters"].items():
                    predicted_parameters.add((i, op["canonical_action"], name, p["source"], json.dumps(p["value"], sort_keys=True)))
                for edge in op["dependencies"]:
                    predicted_relations.add((i, edge["operation_index"], edge["relation"]))
            parameter_tp += len(predicted_parameters & gold_parameters)
            parameter_predicted += len(predicted_parameters)
            parameter_gold += len(gold_parameters)
            relation_tp += len(predicted_relations & gold_relations)
            relation_predicted += len(predicted_relations)
            relation_gold += len(gold_relations)
    if not samples:
        raise ValueError("Empty validation loader")
    tp = confusion.diag().float()
    support, predicted = confusion.sum(1).float(), confusion.sum(0).float()
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    result = {key: value / samples for key, value in totals.items()}
    result.update({
        "samples": samples, "act_macro_f1": float(f1.mean()), "act_confusion_matrix": confusion.tolist(),
        "act_per_class_recall": {name: float(tp[i] / support[i]) if support[i] else None for i, name in enumerate(r["acts"])},
        "full_frame_exact_match": full / samples, "coverage": accepted / samples,
        "accepted_error_rate": 1 - accepted_correct / accepted if accepted else None,
        "goal_recall_including_presence_misses": goal_correct / gold_ops if gold_ops else None,
        "canonical_action_recall": action_correct / gold_ops if gold_ops else None,
        "action_accuracy_oracle_goal": oracle_actions / oracle_count if oracle_count else None,
        "operation_count_exact_match": count_correct / samples,
        "overflow_recall": overflow_correct / overflow_total if overflow_total else None,
        "parameter_value_precision": parameter_tp / parameter_predicted if parameter_predicted else 0.0,
        "parameter_value_recall": parameter_tp / parameter_gold if parameter_gold else None,
        "parameter_value_f1": 2 * parameter_tp / max(parameter_predicted + parameter_gold, 1),
        "dependency_f1": 2 * relation_tp / max(relation_predicted + relation_gold, 1),
        "ood_false_accept": unknown_accepted / unknown_total if unknown_total else None,
        "ood_false_reject": known_rejected / known_total if known_total else None,
        "act_calibration": calibration_metrics(torch.cat(probabilities), torch.cat(act_targets)),
    })
    return result


def _atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, epoch=-1,
                    metrics=None, best_metric=None, data_splits=None):
    validate_calibration(model.calibration, model)
    state = {
        "checkpoint_version": 1, "model_state": model.state_dict(), "config": copy.deepcopy(model.config),
        "config_hash": fingerprint(model.config), "ontology_manifest": copy.deepcopy(model.registry),
        "manifest_hash": fingerprint(model.registry), "ontology_version": model.config["ontology_version"],
        "tokenizer": model.tokenizer.to_dict(), "tokenizer_hash": model.tokenizer.hash,
        "normalizer_version": model.config["normalizer_version"], "calibration": copy.deepcopy(model.calibration),
        "epoch": epoch, "metrics": metrics or {}, "best_metric": best_metric,
        "data_splits": data_splits or {}, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    for key, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None:
            state[key] = obj.state_dict()
    _atomic_save(state, path)


def row_transfer_plan(old_labels, new_labels):
    if len(set(old_labels)) != len(old_labels) or len(set(new_labels)) != len(new_labels):
        raise ValueError("Duplicate labels")
    if not set(old_labels) <= set(new_labels):
        raise ValueError("Expansion cannot silently remove labels")
    positions = {label: i for i, label in enumerate(new_labels)}
    return [(i, positions[label]) for i, label in enumerate(old_labels)]


@torch.no_grad()
def transfer_linear(old_head, new_head, old_labels, new_labels):
    if old_head.weight.shape != (len(old_labels), old_head.in_features) or new_head.weight.shape != (len(new_labels), old_head.in_features):
        raise ValueError("Head/manifest or hidden size mismatch")
    if (old_head.bias is None) != (new_head.bias is None):
        raise ValueError("Bias layout mismatch")
    for old_i, new_i in row_transfer_plan(old_labels, new_labels):
        new_head.weight[new_i].copy_(old_head.weight[old_i])
        if old_head.bias is not None:
            new_head.bias[new_i].copy_(old_head.bias[old_i])


@torch.no_grad()
def transfer_embedding(old_embedding, new_embedding, old_keys, new_keys):
    if old_embedding.weight.shape != (len(old_keys), old_embedding.embedding_dim) or new_embedding.weight.shape != (len(new_keys), old_embedding.embedding_dim):
        raise ValueError("Embedding/manifest mismatch")
    for old_i, new_i in row_transfer_plan(old_keys, new_keys):
        new_embedding.weight[new_i].copy_(old_embedding.weight[old_i])


def expand_from_checkpoint(checkpoint_path, new_config, device="cpu"):
    """Append labels, rebuild masks and return new model + transfer report.

    No optimizer is reused. Tokenizer/backbone/relation capacity migrations are
    intentionally separate operations, not guessed from matching tensor shapes.
    """
    old = load_for_inference(checkpoint_path, "cpu")
    validate_nlu_config(new_config)
    for key in ("architecture", "config_version", "normalizer_version", "tokenizer", "semantic_core", "operation_decoder", "relations"):
        if new_config[key] != old.config[key]:
            raise ValueError(f"Expansion cannot change {key}; explicit architecture migration required")
    registry = build_registry(new_config, previous=old.registry)
    if registry != old.registry and new_config["ontology_version"] == old.config["ontology_version"]:
        raise ValueError("Ontology changes require a new ontology_version")
    new = build_model(new_config, old.tokenizer, registry)
    old_parameters, new_parameters = dict(old.named_parameters()), dict(new.named_parameters())
    dynamic = ("act.", "goal.", "action.", "slot_embedding.", "state_path.", "enum_heads.")
    copied = []
    with torch.no_grad():
        for name, parameter in old_parameters.items():
            if name.startswith(dynamic):
                continue
            if name not in new_parameters or new_parameters[name].shape != parameter.shape:
                raise ValueError(f"Incompatible shared parameter {name}")
            new_parameters[name].copy_(parameter)
            copied.append(name)
        for name, labels in (("act", "acts"), ("goal", "domains"), ("action", "actions")):
            transfer_linear(getattr(old, name), getattr(new, name), old.registry[labels], registry[labels])
        old_keys, new_keys = [x["key"] for x in old.registry["slots"]], [x["key"] for x in registry["slots"]]
        transfer_embedding(old.slot_embedding, new.slot_embedding, old_keys, new_keys)
        if old.state_path is not None:
            transfer_linear(old.state_path, new.state_path, old.registry["state_paths"], registry["state_paths"])
        for old_i, new_i in row_transfer_plan(old_keys, new_keys):
            if str(old_i) in old.enum_heads:
                new.enum_heads[str(new_i)].load_state_dict(old.enum_heads[str(old_i)].state_dict(), strict=True)
    new.calibration = {}  # Softmax denominators changed; refit after fine-tuning.
    report = {"shared_parameters_copied": copied, "optimizer_reset_required": True,
              "added": {key: [x for x in registry[key] if x not in old.registry[key]] for key in ("acts", "domains", "actions", "state_paths")},
              "added_slots": [x for x in new_keys if x not in old_keys]}
    return new.to(device), report


def fit_temperature(logits, targets, max_iter=100):
    """Post-hoc scalar optimization only; never changes model weights."""
    logits, targets = logits.detach().float().cpu(), targets.detach().long().cpu()
    if logits.ndim != 2 or targets.shape != (len(logits),) or not len(targets):
        raise ValueError("Calibration requires nonempty logits [N,C] and targets [N]")
    if not torch.isfinite(logits.gather(1, targets[:, None])).all():
        raise ValueError("Gold class masked in calibration data")
    # Fully masked classes remain -inf; their temperature derivative must be
    # zero rather than -inf/T, otherwise calibration gradients become NaN.
    finite = torch.isfinite(logits)
    safe_logits = logits.masked_fill(~finite, 0)
    log_t = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_t.clamp(-5, 5).exp()
        scaled = (safe_logits / temperature).masked_fill(~finite, float("-inf"))
        loss = F.cross_entropy(scaled, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().clamp(-5, 5).exp())


def fit_value_calibrator(features, labels, max_iter=100):
    """Rows: [raw_value_score, source_score, presence, goal_conf, action_conf].

    Labels must compare extractor predictions with gold action/source/value on
    held-out/cross-fitted data, NOT be constant 1 for annotated samples.
    """
    features, labels = torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.float32)
    if features.ndim != 2 or features.size(1) != 5 or labels.shape != (features.size(0),):
        raise ValueError("Expected value calibration features [N,5], labels [N]")
    if not torch.isfinite(features).all() or not ((labels == 0) | (labels == 1)).all() or labels.unique().numel() != 2:
        raise ValueError("Value calibration requires finite features and both correctness classes")
    head = nn.Linear(5, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(features).squeeze(-1), labels) + 1e-4 * head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"feature_version": 1, "weights": head.weight.detach()[0].tolist(), "bias": float(head.bias.detach()[0])}


def fit_calibration(model, loader, calibration_identity, previous_splits):
    """Freeze extractor; fit scores on held-out predictions versus ground truth.

    Value targets include selected action, source AND value correctness. Parsing
    failures remain rejected by the validator, not accepted calibration examples.
    Metrics here describe the calibration set, not an independent test estimate.
    """
    if not previous_splits.get("train") or not previous_splits.get("validation"):
        raise ValueError("Calibration requires checkpoint split provenance")
    for identity in previous_splits.values():
        assert_disjoint(identity, calibration_identity)
    model.eval()
    device = next(model.parameters()).device
    store = {name: [[], []] for name in ("act", "goal")}
    operation_logits = []
    with torch.no_grad():
        for batch in loader:
            inputs, targets = _device_batch(batch, device)
            output = model(**inputs)
            store["act"][0].append(output["act_logits"].cpu())
            store["act"][1].append(targets["act_y"].cpu())
            valid = targets["goal_y"].ne(IGNORE)
            store["goal"][0].append(output["goal_logits"][valid].cpu())
            store["goal"][1].append(targets["goal_y"][valid].cpu())
            operation_logits.append((output["goal_logits"][valid].cpu(), output["action_logits"][valid].cpu(),
                                     targets["goal_y"][valid].cpu(), targets["action_y"][valid].cpu()))
    temperatures, metrics = {}, {}
    for name, (logit_parts, target_parts) in store.items():
        logits, targets = torch.cat(logit_parts), torch.cat(target_parts)
        if len(targets):
            temperature = fit_temperature(logits, targets)
            temperatures[name] = temperature
            metrics[name] = {"samples": len(targets), "before": calibration_metrics(logits.softmax(-1), targets),
                             "after": calibration_metrics((logits / temperature).softmax(-1), targets)}
        else:
            metrics[name] = {"samples": 0, "status": "insufficient_data"}
    # Use the production hierarchical pair decoder, not an unrelated goal argmax.
    # Select the conditional calibration cohort after goal temperature fitting,
    # then report pair accuracy on ALL operations after action fitting as well.
    action_logits, action_targets = [], []
    mask_domains = model.action_domain.cpu()
    for goals, actions, goal_y, action_y in operation_logits:
        for goal, action, gold_goal, gold_action in zip(goals, actions, goal_y, action_y):
            domain_id, _, _, _ = decode_goal_action(model, goal.to(device), action.to(device), temperatures)
            if domain_id == int(gold_goal):
                action_logits.append(action.masked_fill(mask_domains != domain_id, float("-inf")))
                action_targets.append(int(gold_action))
    if action_targets:
        logits, targets = torch.stack(action_logits), torch.tensor(action_targets)
        temperature = fit_temperature(logits, targets)
        temperatures["action"] = temperature
        metrics["action"] = {"samples": len(targets), "before": calibration_metrics(logits.softmax(-1), targets),
                             "after": calibration_metrics((logits / temperature).softmax(-1), targets)}
    else:
        metrics["action"] = {"samples": 0, "status": "insufficient_correct_domain_predictions"}
    pair_correct = pair_total = 0
    for goals, actions, goal_y, action_y in operation_logits:
        for goal, action, gold_goal, gold_action in zip(goals, actions, goal_y, action_y):
            domain_id, action_id, _, _ = decode_goal_action(model, goal.to(device), action.to(device), temperatures)
            pair_correct += domain_id == int(gold_goal) and action_id == int(gold_action)
            pair_total += 1
    model.calibration = {"temperatures": temperatures, "tokenizer_hash": model.tokenizer.hash,
                         "manifest_hash": fingerprint(model.registry), "method": "temperature_scaling_v1",
                         "metrics": metrics, "action_fit_condition": "correct_pair_selected_domain_before_action_fit",
                         "domain_action_exact_all": pair_correct / pair_total if pair_total else None}
    features, labels = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, _ = _device_batch(batch, device)
            output = model(**inputs)
            for b, (row, encoded) in enumerate(zip(batch["rows"], batch["encoded"])):
                prediction = decode_prediction(model, output, encoded, b, include_value_features=True)
                for op in prediction["operations"]:
                    i = op["operation_index"]
                    gold = row["operations"][i] if i < len(row["operations"]) else None
                    action_correct = gold is not None and _canonical(gold) == op["canonical_action"]
                    for name, parameter in op["parameters"].items():
                        target = gold.get("parameters", {}).get(name) if gold is not None else None
                        correct = (action_correct and target is not None and target["source"] == parameter["source"]
                                   and target["value"] == parameter["value"])
                        features.append(parameter["calibration_features"])
                        labels.append(int(correct))
    if len(set(labels)) == 2:
        model.calibration["value"] = fit_value_calibrator(features, labels)
        metrics["value"] = {"samples": len(labels), "correct": sum(labels), "status": "fitted"}
    else:
        metrics["value"] = {"samples": len(labels), "status": "insufficient_correct_and_incorrect_candidates"}
    return model.calibration


def fit(config, train_manifest, validation_manifest, output_dir, device="cpu", resume=None, expand=None):
    if resume and expand:
        raise ValueError("Resume and expansion are mutually exclusive")
    validate_nlu_config(config)
    t = config["training"]
    torch.manual_seed(t["seed"])
    random.seed(t["seed"])
    train_rows, dev_rows = read_manifest(train_manifest), read_manifest(validation_manifest)
    splits = {"train": split_identity(train_rows), "validation": split_identity(dev_rows)}
    assert_disjoint(splits["train"], splits["validation"])
    saved, transfer_report = None, None
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["config"] != config or saved.get("data_splits") != splits:
            raise ValueError("Resume needs exactly the same config and dataset splits")
        model = load_for_inference(resume, device)
    elif expand:
        model, transfer_report = expand_from_checkpoint(expand, config, device)
        old_saved = torch.load(expand, map_location="cpu", weights_only=True)
        # Old evaluation data must not become training replay.
        for split in ("validation", "calibration"):
            if split in old_saved.get("data_splits", {}):
                assert_disjoint(splits["train"], old_saved["data_splits"][split])
    else:
        tokenizer = CharacterTokenizer.fit([r["text"] for r in train_rows], config["tokenizer"]["max_length"])
        model = build_model(config, tokenizer).to(device)
    model.calibration = {}  # Training invalidates old calibration, even after resume.
    train = NLUDataset(train_rows, config, model.registry, model.tokenizer)
    dev = NLUDataset(dev_rows, config, model.registry, model.tokenizer)
    train_loader = DataLoader(train, batch_size=t["batch_size"], shuffle=True, collate_fn=collate_nlu, num_workers=t["num_workers"])
    dev_loader = DataLoader(dev, batch_size=t["batch_size"], collate_fn=collate_nlu, num_workers=t["num_workers"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] == "fp16")
    start, best = 0, -math.inf
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start, best = saved["epoch"] + 1, saved["best_metric"]
    output_dir = Path(output_dir)
    save_config(config, output_dir / "config.json")
    if transfer_report:
        with (output_dir / "expansion.json").open("w", encoding="utf-8") as stream:
            json.dump(transfer_report, stream, ensure_ascii=False, indent=2)
    for epoch in range(start, t["max_epochs"]):
        training = train_one_epoch(model, train_loader, optimizer, scaler)
        metrics = validate(model, dev_loader)
        scheduler.step()
        score = metrics["full_frame_exact_match"]
        improved = score > best
        best = max(best, score)
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, metrics, best, splits)
        if improved:
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, metrics, best, splits)
        record = {"epoch": epoch, "train": training, "validation": metrics}
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--write-default-config")
    parser.add_argument("--train-manifest")
    parser.add_argument("--validation-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume")
    group.add_argument("--expand-from")
    group.add_argument("--calibrate-checkpoint")
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--calibrated-output")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else create_config()
    if args.write_default_config:
        save_config(config, args.write_default_config)
        return
    if args.calibrate_checkpoint:
        if not args.calibration_manifest or not args.calibrated_output:
            parser.error("Calibration requires --calibration-manifest and --calibrated-output")
        model = load_for_inference(args.calibrate_checkpoint, args.device)
        saved = torch.load(args.calibrate_checkpoint, map_location="cpu", weights_only=True)
        rows = read_manifest(args.calibration_manifest)
        dataset = NLUDataset(rows, model.config, model.registry, model.tokenizer)
        loader = DataLoader(dataset, batch_size=model.config["training"]["batch_size"], collate_fn=collate_nlu)
        identity = split_identity(rows)
        result = fit_calibration(model, loader, identity, saved.get("data_splits", {}))
        splits = copy.deepcopy(saved["data_splits"])
        splits["calibration"] = identity
        save_checkpoint(args.calibrated_output, model, metrics={"calibration": result}, data_splits=splits)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not args.train_manifest or not args.validation_manifest or not args.output_dir:
        parser.error("Training requires --train-manifest, --validation-manifest and --output-dir")
    fit(config, args.train_manifest, args.validation_manifest, args.output_dir, args.device, args.resume, args.expand_from)


if __name__ == "__main__":
    main()
