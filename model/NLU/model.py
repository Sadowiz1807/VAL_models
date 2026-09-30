"""Text-only semantic model, character tokenizer, typed decoding and safe output.

No external weights are downloaded. Character vocabulary is fitted on TRAIN text
only, persisted with the checkpoint and never silently refitted at inference.
Offsets are Unicode code points on NFC text. Runtime, not this module, resolves
state/context, app aliases and command allowlists or executes tools.
"""
from __future__ import annotations

import copy
import math
import re
import unicodedata

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

if __package__:
    from .config import create_config, validate_nlu_config, build_registry, validate_registry, fingerprint, SOURCES
else:
    from config import create_config, validate_nlu_config, build_registry, validate_registry, fingerprint, SOURCES


def normalize_text(text):
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    return unicodedata.normalize("NFC", text)


class CharacterTokenizer:
    SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    pad_id, unk_id, bos_id, eos_id = range(4)

    def __init__(self, tokens, max_length=512):
        if list(tokens[:4]) != self.SPECIALS or len(set(tokens)) != len(tokens):
            raise ValueError("Invalid tokenizer vocabulary/special IDs")
        if any(not isinstance(x, str) or len(x) != 1 or normalize_text(x) != x for x in tokens[4:]):
            raise ValueError("Vocabulary entries must be NFC characters")
        if type(max_length) is not int or max_length < 3:
            raise ValueError("Invalid tokenizer max_length")
        self.tokens, self.max_length = list(tokens), max_length
        self.ids = {token: i for i, token in enumerate(tokens)}

    @classmethod
    def fit(cls, train_texts, max_length=512):
        chars = sorted(set("".join(normalize_text(text) for text in train_texts)))
        if not chars:
            raise ValueError("Cannot fit tokenizer on an empty corpus")
        return cls(cls.SPECIALS + chars, max_length)

    def __len__(self):
        return len(self.tokens)

    def to_dict(self):
        return {"type": "unicode_character_v1", "normalizer_version": "unicode_nfc_v1",
                "tokens": self.tokens, "max_length": self.max_length}

    @classmethod
    def from_dict(cls, state):
        if state.get("type") != "unicode_character_v1" or state.get("normalizer_version") != "unicode_nfc_v1":
            raise ValueError("Unsupported tokenizer profile")
        return cls(state["tokens"], state["max_length"])

    @property
    def hash(self):
        return fingerprint(self.to_dict())

    def encode(self, text):
        text = normalize_text(text)
        if not text.strip():
            raise ValueError("Empty text")
        if len(text) + 2 > self.max_length:
            raise ValueError("Input exceeds max_length; truncation is forbidden")
        ids = [self.bos_id] + [self.ids.get(char, self.unk_id) for char in text] + [self.eos_id]
        return {"text": text, "input_ids": ids,
                "offsets": [(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)],
                "span_valid_mask": [False] + [True] * len(text) + [False],
                "unknown_count": ids.count(self.unk_id)}

    def batch_encode(self, texts, device="cpu"):
        encoded = [self.encode(text) for text in texts]
        if not encoded:
            raise ValueError("Empty text batch")
        width = max(len(x["input_ids"]) for x in encoded)
        ids = torch.full((len(encoded), width), self.pad_id, dtype=torch.long, device=device)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        span = torch.zeros_like(valid)
        for i, item in enumerate(encoded):
            size = len(item["input_ids"])
            ids[i, :size] = torch.tensor(item["input_ids"], device=device)
            valid[i, :size] = True
            span[i, :size] = torch.tensor(item["span_valid_mask"], device=device)
        return {"input_ids": ids, "attention_mask": valid, "span_valid_mask": span, "encoded": encoded}


def _schema_masks(registry):
    s, states, contexts = len(registry["slots"]), len(registry["state_paths"]), len(registry["context_types"])
    source = torch.zeros(s, 3, dtype=torch.bool)
    state = torch.zeros(s, states, dtype=torch.bool)
    context = torch.zeros(s, contexts, dtype=torch.bool)
    for i, slot in enumerate(registry["slots"]):
        p = slot["schema"]
        source[i, 0] = p["type"] not in ("STATE_REFERENCE", "CONTEXT_REFERENCE")
        for name in p.get("state_reference_paths", []):
            state[i, registry["state_paths"].index(name)] = True
            source[i, 1] = True
        for name in p.get("context_reference_types", []):
            context[i, registry["context_types"].index(name)] = True
            source[i, 2] = True
    return source, state, context


class TextSemanticModel(nn.Module):
    def __init__(self, config, registry, tokenizer):
        super().__init__()
        validate_nlu_config(config)
        validate_registry(config, registry)
        if tokenizer.max_length != config["tokenizer"]["max_length"]:
            raise ValueError("Tokenizer/config max_length mismatch")
        self.config, self.registry = copy.deepcopy(config), copy.deepcopy(registry)
        self.tokenizer, self.calibration = tokenizer, {}
        c, decoder = config["semantic_core"], config["operation_decoder"]
        d, k, s = c["d_model"], decoder["max_operations"], len(registry["slots"])
        self.d, self.k, self.max_length = d, k, tokenizer.max_length
        self.token_embedding = nn.Embedding(len(tokenizer), d, padding_idx=tokenizer.pad_id)
        self.position_embedding = nn.Embedding(self.max_length, d)
        # Instantiate each layer separately: no identical cloned initialization.
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=c["attention_heads"], dim_feedforward=c["d_ff"],
                                       dropout=c["dropout"], batch_first=True, norm_first=True)
            for _ in range(c["layers"])
        ])
        self.op_queries = nn.Parameter(torch.randn(k, d) * 0.02)
        self.op_attention = nn.MultiheadAttention(d, decoder["attention_heads"], dropout=decoder["dropout"], batch_first=True)
        self.op_norm = nn.LayerNorm(d)
        self.act = nn.Linear(d, len(registry["acts"]))
        self.turn = nn.Linear(d, len(registry["turn_relations"]))
        self.context_required = nn.Linear(d, 1)
        self.turn_context = nn.Linear(d, len(registry["context_types"]))
        self.ood = nn.Linear(d, 1)
        # Classes 0..K plus K+1 = overflow, supervised with actual overflow data.
        self.count = nn.Linear(d, k + 2)
        self.presence = nn.Linear(d, 1)
        self.goal = nn.Linear(d, len(registry["domains"]))
        self.action = nn.Linear(d, len(registry["actions"]))
        self.slot_embedding = nn.Embedding(s, d)
        self.slot_fuse = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.slot_presence = nn.Linear(d, 1)
        self.source = nn.Linear(d, 3)
        self.start_query, self.end_query = nn.Linear(d, d), nn.Linear(d, d)
        self.state_path = nn.Linear(d, len(registry["state_paths"])) if registry["state_paths"] else None
        self.slot_context = nn.Linear(d, len(registry["context_types"]))
        self.enum_heads = nn.ModuleDict({str(i): nn.Linear(d, len(slot["schema"]["values"]))
                                       for i, slot in enumerate(registry["slots"]) if slot["schema"]["type"] == "ENUM"})
        self.boolean = nn.Linear(d, 1)
        self.op_relation = nn.Linear(2 * d, len(registry["operation_relations"]))
        self.register_buffer("action_domain", torch.tensor(registry["action_to_domain"], dtype=torch.long))
        self.register_buffer("slot_action", torch.tensor([x["action_id"] for x in registry["slots"]], dtype=torch.long))
        source, state, context = _schema_masks(registry)
        self.register_buffer("source_allowed", source)
        self.register_buffer("state_allowed", state)
        self.register_buffer("context_allowed", context)

    def forward(self, input_ids, attention_mask, span_valid_mask=None):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or attention_mask.shape != input_ids.shape:
            raise ValueError("Expected int64 input_ids and mask [B,L]")
        valid = attention_mask.bool()
        b, length = input_ids.shape
        if length > self.max_length or (~valid.any(1)).any():
            raise ValueError("Input too long or contains empty sequences")
        if span_valid_mask is None:
            span_valid_mask = valid & input_ids.ne(self.tokenizer.pad_id) & input_ids.ne(self.tokenizer.bos_id) & input_ids.ne(self.tokenizer.eos_id)
        if span_valid_mask.shape != valid.shape or (span_valid_mask & ~valid).any():
            raise ValueError("Invalid span mask")
        pos = torch.arange(length, device=input_ids.device)
        h = self.token_embedding(input_ids) + self.position_embedding(pos)[None]
        for layer in self.encoder:
            if self.training and self.config["semantic_core"]["gradient_checkpointing"]:
                h = checkpoint(layer, h, src_key_padding_mask=~valid, use_reentrant=False)
            else:
                h = layer(h, src_key_padding_mask=~valid)
        h = h.masked_fill(~valid[..., None], 0)
        turn = h.sum(1) / valid.sum(1, keepdim=True)
        q = self.op_queries[None].expand(b, -1, -1)
        attended, _ = self.op_attention(q, h, h, key_padding_mask=~valid, need_weights=False)
        op = self.op_norm(q + attended)
        slot = self.slot_fuse(op[:, :, None] + self.slot_embedding.weight[None, None])
        start = torch.einsum("bksd,bld->bksl", self.start_query(slot), h) / math.sqrt(self.d)
        end = torch.einsum("bksd,bld->bksl", self.end_query(slot), h) / math.sqrt(self.d)
        start = start.masked_fill(~span_valid_mask[:, None, None], float("-inf"))
        end = end.masked_fill(~span_valid_mask[:, None, None], float("-inf"))
        left = op[:, :, None].expand(-1, -1, self.k, -1)
        right = op[:, None].expand(-1, self.k, -1, -1)
        return {
            "token_states": h, "turn_state": turn, "operation_states": op,
            "act_logits": self.act(turn), "turn_relation_logits": self.turn(turn),
            "context_required_logit": self.context_required(turn).squeeze(-1),
            "context_reference_logits": self.turn_context(turn), "ood_logit": self.ood(turn).squeeze(-1),
            "count_logits": self.count(turn), "presence_logits": self.presence(op).squeeze(-1),
            "goal_logits": self.goal(op), "action_logits": self.action(op),
            "slot_presence_logits": self.slot_presence(slot).squeeze(-1), "source_logits": self.source(slot),
            "span_start_logits": start, "span_end_logits": end,
            "state_path_logits": self.state_path(slot) if self.state_path is not None else slot.new_empty((*slot.shape[:-1], 0)),
            "slot_context_logits": self.slot_context(slot),
            "enum_logits": {key: head(slot[:, :, int(key)]) for key, head in self.enum_heads.items()},
            "boolean_logits": self.boolean(slot).squeeze(-1),
            "operation_relation_logits": self.op_relation(torch.cat([left, right], -1)),
        }

    def mask_actions(self, logits, goal_ids):
        allowed = self.action_domain[None, None] == goal_ids[..., None]
        return logits.masked_fill(~allowed, float("-inf"))

    @torch.inference_mode()
    def predict(self, text, request_id=None):
        self.eval()
        base = {"schema_version": self.config["schema_version"], "request_id": request_id}
        try:
            batch = self.tokenizer.batch_encode([text], next(self.parameters()).device)
        except (ValueError, TypeError) as error:
            return {**base, "status": "invalid_text", "error": str(error), "operations": [], "ready_for_harness": False}
        output = self(batch["input_ids"], batch["attention_mask"], batch["span_valid_mask"])
        try:
            result = decode_prediction(self, output, batch["encoded"][0])
        except (ValueError, FloatingPointError) as error:
            return {**base, "status": "invalid_model_output", "error": str(error),
                    "operations": [], "ready_for_harness": False}
        result.update(base)
        return result


def selected_class_confidence(logits, selected_ids, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return (logits.float() / temperature).softmax(-1).gather(-1, selected_ids[..., None]).squeeze(-1)


def _selected(logits, temperature=1.0, allowed=None):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid temperature")
    logits = logits.float() / temperature
    if allowed is not None:
        logits = logits.masked_fill(~allowed, float("-inf"))
    if torch.isnan(logits).any() or not torch.isfinite(logits).any():
        raise ValueError("No valid finite class")
    probabilities = logits.softmax(-1)
    index = int(probabilities.argmax())
    return index, float(probabilities[index])


def decode_span(start, end, encoded, max_span_length):
    length = len(encoded["input_ids"])
    start, end = start[:length].float(), end[:length].float()
    valid = torch.tensor(encoded["span_valid_mask"], device=start.device)
    start, end = start.masked_fill(~valid, float("-inf")), end.masked_fill(~valid, float("-inf"))
    if not valid.any():
        raise ValueError("No valid span positions")
    positions = torch.arange(length, device=start.device)
    allowed = ((positions[:, None] <= positions[None, :]) &
               (positions[None, :] - positions[:, None] < max_span_length) & valid[:, None] & valid[None, :])
    scores = (start.log_softmax(-1)[:, None] + end.log_softmax(-1)[None]).masked_fill(~allowed, float("-inf"))
    flat = int(scores.argmax())
    i, j = divmod(flat, length)
    a, z = encoded["offsets"][i][0], encoded["offsets"][j][1]
    score = float(scores[i, j].exp())  # Ranking score, not calibrated correctness.
    return encoded["text"][a:z], [a, z], score


_DIGITS = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
           "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9}


def _under_thousand(tokens):
    if not tokens:
        raise ValueError("Missing number")
    if len(tokens) == 1 and tokens[0] in _DIGITS:
        return _DIGITS[tokens[0]]
    result = 0
    if len(tokens) >= 2 and tokens[1] == "trăm" and tokens[0] in _DIGITS:
        result, tokens = _DIGITS[tokens[0]] * 100, tokens[2:]
        if not tokens:
            return result
        if tokens[0] in ("linh", "lẻ"):
            tokens = tokens[1:]
            if len(tokens) != 1 or tokens[0] not in _DIGITS:
                raise ValueError("Invalid Vietnamese number")
            return result + _DIGITS[tokens[0]]
    if tokens[0] == "mười":
        tens, rest = 10, tokens[1:]
    elif len(tokens) >= 2 and tokens[0] in _DIGITS and _DIGITS[tokens[0]] >= 2 and tokens[1] == "mươi":
        tens, rest = _DIGITS[tokens[0]] * 10, tokens[2:]
    else:
        raise ValueError("Unsupported or ambiguous Vietnamese number")
    if not rest:
        return result + tens
    if len(rest) == 1 and rest[0] in _DIGITS:
        return result + tens + _DIGITS[rest[0]]
    raise ValueError("Invalid Vietnamese number")


def parse_number(text):
    """Strict supported grammar; no clipping, substring salvage or guessed units."""
    value = normalize_text(text).strip().lower()
    for suffix in ("phần trăm", "%", "bước", "nấc", "lần"):
        if value.endswith(suffix):
            value = value[:-len(suffix)].strip()
            break
    if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", value):
        number = float(value.replace(",", "."))
    else:
        tokens = value.split()
        sign = -1 if tokens and tokens[0] == "âm" else 1
        if sign < 0:
            tokens = tokens[1:]
        decimal = tokens.index("phẩy") if "phẩy" in tokens else len(tokens)
        integer_tokens, fractional = tokens[:decimal], tokens[decimal + 1:]
        scales = {"tỷ": 1_000_000_000, "triệu": 1_000_000, "nghìn": 1000, "ngàn": 1000}
        integer, segment, last_scale = 0, [], float("inf")
        for token in integer_tokens:
            if token in scales:
                scale = scales[token]
                if scale >= last_scale:
                    raise ValueError("Invalid scale order")
                integer += _under_thousand(segment) * scale
                segment, last_scale = [], scale
            else:
                segment.append(token)
        if segment:
            integer += _under_thousand(segment)
        elif not integer_tokens:
            raise ValueError("Missing numeric value")
        if decimal < len(tokens):
            if not fractional or any(token not in _DIGITS for token in fractional):
                raise ValueError("Invalid fractional digits")
            fraction = float("0." + "".join(str(_DIGITS[token]) for token in fractional))
        else:
            fraction = 0
        number = sign * (integer + fraction)
    if not math.isfinite(number):
        raise ValueError("Nonfinite numeric value")
    return int(number) if float(number).is_integer() else number


def validate_parameter_value(value, schema, source):
    if source == "STATE_REFERENCE":
        if value not in schema.get("state_reference_paths", []):
            raise ValueError("State reference is not allowed for slot")
        return
    if source == "CONTEXT_REFERENCE":
        if value not in schema.get("context_reference_types", []):
            raise ValueError("Context reference is not allowed for slot")
        return
    if source != "INPUT_SPAN":
        raise ValueError("Unknown value source")
    kind = schema["type"]
    if kind == "NUMBER":
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Expected finite number (not bool)")
        if schema.get("integer") and type(value) is not int:
            raise ValueError("Expected integer")
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError("Number outside schema range")
    elif kind in ("ENTITY", "FREE_TEXT"):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Empty entity/free text")
    elif kind == "ENUM":
        if value not in schema["values"]:
            raise ValueError("Invalid enum value")
    elif kind == "BOOLEAN":
        if type(value) is not bool:
            raise ValueError("Expected boolean")
    else:
        raise ValueError("Reference slot cannot use INPUT_SPAN")


def validate_tab_target(parameters):
    present = [name for name in ("tab_index", "tab_query", "tab_reference") if parameters.get(name) is not None]
    if len(present) > 1:
        raise ValueError("Conflicting tab targets")
    index = parameters.get("tab_index")
    if index is not None and (type(index) is not int or index < 1):
        raise ValueError("tab_index must be a positive integer")
    return {"needs_context": not present}


def validate_output(payload, registry):
    """Return schema errors without resolving references or altering predictions."""
    errors = []
    if payload.get("act") not in registry["acts"]:
        errors.append("invalid_act")
    operations = payload.get("operations", [])
    indices = [op["operation_index"] for op in operations]
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        errors.append("invalid_operation_order")
    if payload.get("act") == "EXECUTE" and not operations:
        errors.append("execute_without_operation")
    for op in operations:
        canonical = op["canonical_action"]
        if canonical not in registry["actions"]:
            errors.append("unknown_action")
            continue
        action_id = registry["actions"].index(canonical)
        if op["goal"] != registry["domains"][registry["action_to_domain"][action_id]] or op["action"] != canonical.split(".", 1)[1]:
            errors.append("goal_action_mismatch")
        specs = {x["name"]: x["schema"] for x in registry["slots"] if x["action_id"] == action_id}
        parameters = op["parameters"]
        for name, p in parameters.items():
            if name not in specs:
                errors.append(f"unknown_parameter:{name}")
                continue
            try:
                validate_parameter_value(p["value"], specs[name], p["source"])
            except (ValueError, KeyError) as error:
                errors.append(f"invalid_parameter:{name}:{error}")
        for name, spec in specs.items():
            if spec["required"] and name not in parameters:
                errors.append(f"missing_required:{op['operation_index']}:{name}")
        constraints = registry["action_constraints"][canonical]
        for group in constraints.get("mutually_exclusive", []):
            if sum(name in parameters for name in group) > 1:
                errors.append("mutually_exclusive_parameters")
        for edge in op["dependencies"]:
            if edge["operation_index"] not in indices or edge["operation_index"] >= op["operation_index"]:
                errors.append("invalid_dependency")
            if edge["relation"] not in registry["operation_relations"] or edge["relation"] == "NONE":
                errors.append("invalid_dependency_type")
    return errors


def validate_calibration(calibration, model):
    if not calibration:
        return
    if calibration.get("tokenizer_hash") != model.tokenizer.hash or calibration.get("manifest_hash") != fingerprint(model.registry):
        raise ValueError("Calibration metadata mismatch")
    for name, value in calibration.get("temperatures", {}).items():
        if name not in ("act", "goal", "action") or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid calibration temperature")
    value = calibration.get("value")
    if value is not None:
        if value.get("feature_version") != 1 or len(value.get("weights", [])) != 5:
            raise ValueError("Invalid value calibrator")
        if not all(math.isfinite(x) for x in value["weights"] + [value["bias"]]):
            raise ValueError("Nonfinite value calibrator")


def decode_goal_action(model, goal_logits, action_logits, temperatures=None):
    """Rank all domain/action pairs; return scores for the selected labels."""
    temperatures = temperatures or {}
    domain_logp = (goal_logits.float() / temperatures.get("goal", 1.0)).log_softmax(-1)
    pair_scores = []
    for domain_id in range(len(model.registry["domains"])):
        masked = action_logits.float().masked_fill(model.action_domain != domain_id, float("-inf"))
        conditional = (masked / temperatures.get("action", 1.0)).log_softmax(-1)
        pair_scores.append(domain_logp[domain_id] + conditional)
    pairs = torch.stack(pair_scores)
    flat = int(pairs.argmax())
    domain_id, action_id = divmod(flat, len(model.registry["actions"]))
    return (domain_id, action_id, float(domain_logp[domain_id].exp()),
            float((pairs[domain_id, action_id] - domain_logp[domain_id]).exp()))


def decode_prediction(model, output, encoded, batch_index=0, include_value_features=False):
    for name, tensor in output.items():
        tensors = tensor.values() if isinstance(tensor, dict) else (tensor,)
        for value in tensors:
            if name in ("span_start_logits", "span_end_logits"):
                invalid = torch.isnan(value).any() or torch.isposinf(value).any()
            else:
                invalid = not torch.isfinite(value).all()
            if invalid:
                raise FloatingPointError(f"Nonfinite output: {name}")
    r, config, b = model.registry, model.config, batch_index
    validate_calibration(model.calibration, model)
    temps = model.calibration.get("temperatures", {})
    thresholds = config["thresholds"]
    act_id, act_conf = _selected(output["act_logits"][b], temps.get("act", 1.0))
    turn_id, _ = _selected(output["turn_relation_logits"][b])
    context_required = bool(output["context_required_logit"][b].sigmoid() >= 0.5)
    context_id, _ = _selected(output["context_reference_logits"][b])
    count_id, _ = _selected(output["count_logits"][b])
    ood = float(output["ood_logit"][b].sigmoid())
    payload = {
        "schema_version": config["schema_version"], "text": encoded["text"],
        "act": r["acts"][act_id], "confidence": act_conf,
        "confidence_status": "calibrated" if "act" in temps else "uncalibrated",
        "turn_relation": r["turn_relations"][turn_id], "context_required": context_required,
        "context_reference": r["context_types"][context_id] if context_required else None,
        "ood_score": ood, "predicted_operation_count": count_id if count_id <= model.k else "overflow",
        "operations": [], "issues": [], "requires_context_resolution": context_required,
    }
    issues = payload["issues"]
    if act_conf < thresholds["act_min_confidence"]:
        issues.append("low_act_confidence")
    if ood > thresholds["max_ood_score"]:
        issues.append("out_of_scope_score")
    if encoded["unknown_count"]:
        issues.append("unknown_input_characters")
    if config["decoding"]["require_calibration"] and not all(name in temps for name in ("act", "goal", "action")):
        issues.append("classification_not_calibrated")
    if count_id > model.k:
        issues.append("operation_overflow_split_request")
        payload.update(status="needs_clarification", ready_for_harness=False)
        return payload
    presence = output["presence_logits"][b].sigmoid()
    active = [i for i in range(model.k) if float(presence[i]) >= thresholds["operation_presence_threshold"]]
    if len(active) != count_id:
        issues.append("operation_count_disagreement")
    if active != list(range(len(active))):
        issues.append("noncontiguous_operation_slots")
    for k in active:
        domain_id, action_id, goal_conf, action_conf = decode_goal_action(
            model, output["goal_logits"][b, k], output["action_logits"][b, k], temps)
        canonical = r["actions"][action_id]
        op = {"operation_index": k, "presence_score": float(presence[k]), "goal": r["domains"][domain_id],
              "goal_confidence": goal_conf, "action": canonical.split(".", 1)[1], "canonical_action": canonical,
              "action_confidence": action_conf, "parameters": {}, "missing_required_parameters": [], "dependencies": []}
        if goal_conf < thresholds["goal_min_confidence"] or action_conf < thresholds["action_min_confidence"]:
            issues.append(f"low_goal_action_confidence:{k}")
        for s, slot in enumerate(r["slots"]):
            if slot["action_id"] != action_id:
                continue
            p, name = slot["schema"], slot["name"]
            slot_presence = float(output["slot_presence_logits"][b, k, s].sigmoid())
            if slot_presence < thresholds["slot_presence_threshold"]:
                if p["required"]:
                    op["missing_required_parameters"].append(name)
                continue
            source_id, source_score = _selected(output["source_logits"][b, k, s], allowed=model.source_allowed[s])
            source, span = SOURCES[source_id], None
            try:
                if source == "STATE_REFERENCE":
                    value_id, raw_score = _selected(output["state_path_logits"][b, k, s], allowed=model.state_allowed[s])
                    value = r["state_paths"][value_id]
                    payload["requires_context_resolution"] = True
                elif source == "CONTEXT_REFERENCE":
                    value_id, raw_score = _selected(output["slot_context_logits"][b, k, s], allowed=model.context_allowed[s])
                    value = r["context_types"][value_id]
                    payload["requires_context_resolution"] = True
                else:
                    raw, span, raw_score = decode_span(output["span_start_logits"][b, k, s], output["span_end_logits"][b, k, s],
                                                       encoded, config["decoding"]["max_span_length"])
                    if p["type"] == "NUMBER":
                        value = parse_number(raw)
                    elif p["type"] == "ENUM":
                        value_id, enum_score = _selected(output["enum_logits"][str(s)][b, k])
                        value, raw_score = p["values"][value_id], raw_score * enum_score
                    elif p["type"] == "BOOLEAN":
                        probability = float(output["boolean_logits"][b, k, s].sigmoid())
                        value = probability >= 0.5
                        raw_score *= max(probability, 1 - probability)
                    else:
                        value = raw
                validate_parameter_value(value, p, source)
            except ValueError as error:
                issues.append(f"invalid_parameter:{k}:{name}:{error}")
                if p["required"]:
                    op["missing_required_parameters"].append(name)
                continue
            item = {"value": value, "source": source, "presence_score": slot_presence}
            if span is not None:
                item["span"] = span
            value_cal = model.calibration.get("value")
            features = [raw_score, source_score, slot_presence, goal_conf, action_conf]
            if include_value_features:
                item["calibration_features"] = features
            if value_cal:
                z = sum(x * w for x, w in zip(features, value_cal["weights"])) + value_cal["bias"]
                score = 1 / (1 + math.exp(-max(-60, min(60, z))))
                item.update(confidence=score, confidence_status="calibrated")
            else:
                score = raw_score * source_score
                item.update(value_score=score, confidence_status="uncalibrated")
                if config["decoding"]["require_calibration"]:
                    issues.append(f"value_not_calibrated:{k}:{name}")
            if score < thresholds["parameter_min_confidence"]:
                issues.append(f"low_parameter_score:{k}:{name}")
            op["parameters"][name] = item
        for j in active:
            if j < k:
                relation_id, _ = _selected(output["operation_relation_logits"][b, k, j])
                relation = r["operation_relations"][relation_id]
                if relation != "NONE":
                    op["dependencies"].append({"operation_index": j, "relation": relation})
        constraints = r["action_constraints"][canonical]
        fallback = constraints.get("needs_context_when_absent", [])
        if fallback and not any(name in op["parameters"] for name in fallback):
            payload["requires_context_resolution"] = True
        payload["operations"].append(op)
    if payload["act"] in ("CONFIRM", "CANCEL"):
        payload["requires_context_resolution"] = True
        if payload["context_reference"] is None:
            issues.append("missing_turn_reference")
    if payload["act"] not in ("EXECUTE",) and payload["operations"]:
        issues.append("nonexecute_with_operations_requires_policy")
    issues.extend(validate_output(payload, r))
    payload["issues"] = list(dict.fromkeys(issues))
    if payload["act"] == "UNSUPPORTED" or ood > thresholds["max_ood_score"]:
        status = "unsupported"
    elif issues or payload["act"] == "ASK_CLARIFICATION":
        status = "needs_clarification"
    else:
        status = "ok"
    payload.update(status=status, ready_for_harness=status == "ok")
    # ready_for_harness is never permission to execute; Harness still owns policy.
    return payload


def build_model(config, tokenizer, registry=None):
    return TextSemanticModel(config, build_registry(config) if registry is None else registry, tokenizer)


def load_for_inference(checkpoint_path, device="cpu"):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved.get("checkpoint_version") != 1 or saved.get("config_hash") != fingerprint(saved["config"]):
        raise ValueError("Unsupported/corrupt NLU checkpoint")
    tokenizer = CharacterTokenizer.from_dict(saved["tokenizer"])
    if tokenizer.hash != saved["tokenizer_hash"] or fingerprint(saved["ontology_manifest"]) != saved["manifest_hash"]:
        raise ValueError("Tokenizer/manifest hash mismatch")
    model = build_model(saved["config"], tokenizer, saved["ontology_manifest"])
    for name, buffer in model.named_buffers():
        if name not in saved["model_state"] or not torch.equal(buffer, saved["model_state"][name]):
            raise ValueError(f"Checkpoint schema buffer disagrees with manifest: {name}")
    model.load_state_dict(saved["model_state"], strict=True)
    model.calibration = saved.get("calibration", {})
    validate_calibration(model.calibration, model)
    return model.to(device).eval()


def predict(model, text, request_id=None):
    return model.predict(text, request_id)
