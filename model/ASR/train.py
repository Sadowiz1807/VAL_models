"""ASR JSONL dataset, CTC training, validation and checkpoint CLI.

No training happens on import. Run `python model/ASR/train.py --help`.
Manifests require dataset_id, sample_id, audio_ref, transcript, speaker_id,
session_id. audio_ref is relative to the manifest directory.
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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

if __package__:
    from .config import create_config, load_config, save_config, validate_config, config_hash
    from .model import (build_model, LogMelFrontend, load_audio, encode_utf8_target,
                        min_ctc_frames, decode_ctc_path, normalize_text, encoder_length)
else:
    from config import create_config, load_config, save_config, validate_config, config_hash
    from model import (build_model, LogMelFrontend, load_audio, encode_utf8_target,
                       min_ctc_frames, decode_ctc_path, normalize_text, encoder_length)


class ASRDataset(Dataset):
    def __init__(self, manifest, config):
        validate_config(config)
        self.path, self.config = Path(manifest), copy.deepcopy(config)
        self.frontend = LogMelFrontend(config["audio"])
        with self.path.open(encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        if not self.rows:
            raise ValueError("Empty ASR manifest")
        seen = set()
        for row in self.rows:
            for field in ("dataset_id", "sample_id", "audio_ref", "transcript", "speaker_id", "session_id"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Missing/invalid {field}: {row.get('sample_id')}")
            key = (row["dataset_id"], row["sample_id"])
            if key in seen:
                raise ValueError(f"Duplicate sample {key}")
            seen.add(key)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio, rate = load_audio(self.path.parent / row["audio_ref"], self.config["audio"]["sample_rate"])
        with torch.no_grad():
            features = self.frontend(audio, rate)
        targets = encode_utf8_target(row["transcript"])
        return {"features": features, "targets": targets,
                "sample_id": row["sample_id"], "text": normalize_text(row["transcript"])}

    def audit(self):
        """Run explicitly before training; reject infeasible targets, never hide them."""
        lengths, errors = [], []
        factor = self.config["speech_encoder"]["subsampling_factor"]
        for index, row in enumerate(self.rows):
            try:
                item = self[index]
                length = len(item["features"])
                lengths.append(length)
                if encoder_length(length, factor) < min_ctc_frames(item["targets"]):
                    errors.append(f"{row['sample_id']}: infeasible byte CTC alignment")
            except (ValueError, RuntimeError, OSError) as error:
                lengths.append(0)
                errors.append(f"{row['sample_id']}: {error}")
        return {"samples": len(self), "invalid_count": len(errors),
                "invalid_rate": len(errors) / len(self), "errors": errors, "lengths": lengths}


def assert_disjoint(train: ASRDataset, validation: ASRDataset):
    for field in ("sample_id", "speaker_id", "session_id"):
        left = {(r["dataset_id"], r[field]) for r in train.rows}
        right = {(r["dataset_id"], r[field]) for r in validation.rows}
        if left & right:
            raise ValueError(f"Train/validation leakage in {field}: {list(left & right)[:3]}")
    left_audio = {(train.path.parent / r["audio_ref"]).resolve() for r in train.rows}
    right_audio = {(validation.path.parent / r["audio_ref"]).resolve() for r in validation.rows}
    if left_audio & right_audio:
        raise ValueError("Train/validation reuse the same audio files")
    left_groups = {(r["dataset_id"], r.get("original_audio_id", r["sample_id"])) for r in train.rows}
    right_groups = {(r["dataset_id"], r.get("original_audio_id", r["sample_id"])) for r in validation.rows}
    if left_groups & right_groups:
        raise ValueError("Augmented audio groups leak across splits")


def collate_asr(items):
    if not items or any(not item["targets"] for item in items):
        raise ValueError("ASR batch requires nonempty transcripts")
    features = [item["features"] for item in items]
    targets = [item["targets"] for item in items]
    return {
        "mels": pad_sequence(features, batch_first=True),
        "mel_lengths": torch.tensor([len(x) for x in features], dtype=torch.long),
        "targets": torch.tensor([v for y in targets for v in y], dtype=torch.long),
        "target_lengths": torch.tensor([len(y) for y in targets], dtype=torch.long),
        "min_frames": torch.tensor([min_ctc_frames(y) for y in targets], dtype=torch.long),
        "sample_ids": [item["sample_id"] for item in items],
        "texts": [item["text"] for item in items],
    }


def compute_ctc_loss(output, batch):
    logits, lengths = output["logits"], output["lengths"]
    if (lengths.cpu() < batch["min_frames"].cpu()).any():
        raise ValueError(f"CTC alignment impossible: {batch.get('sample_ids')}")
    targets = batch["targets"]
    if targets.dtype != torch.long or (targets < 1).any() or (targets > 256).any():
        raise ValueError("Invalid byte targets")
    if int(batch["target_lengths"].sum()) != targets.numel():
        raise ValueError("CTC target length mismatch")
    # CPU int64 lengths work with the native CTC path; do not rely on cuDNN's
    # more restrictive equal-length/int32 fast path. CTC itself always uses fp32.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        loss = nn.functional.ctc_loss(
            logits.float().log_softmax(-1).transpose(0, 1), targets.to(logits.device),
            lengths.cpu(), batch["target_lengths"].cpu(), blank=0,
            reduction="mean", zero_infinity=False,
        )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite CTC loss: {batch.get('sample_ids')}")
    return loss


class LengthBucketSampler(Sampler):
    def __init__(self, lengths, batch_size, seed):
        self.lengths, self.batch_size, self.seed, self.epoch = lengths, batch_size, seed, 0

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        batches = []
        window = self.batch_size * 50
        for start in range(0, len(indices), window):
            bucket = sorted(indices[start:start + window], key=lambda i: self.lengths[i])
            batches.extend(bucket[i:i + self.batch_size] for i in range(0, len(bucket), self.batch_size))
        rng.shuffle(batches)
        return iter(batches)


def _amp(config, device):
    setting = config["training"]["amp"]
    if setting == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training requires CUDA")
    return torch.autocast(device_type=device.type, enabled=setting != "off",
                          dtype=torch.float16 if setting == "fp16" else torch.bfloat16)


def train_one_epoch(model, loader, optimizer, scaler=None):
    device = next(model.parameters()).device
    t = model.config["training"]
    scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, count = 0.0, 0
    accumulation = t["gradient_accumulation_steps"]
    for index, batch in enumerate(loader):
        # Correctly normalize a short final accumulation group.
        group_start = (index // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        with _amp(model.config, device):
            output = model(batch["mels"].to(device), batch["mel_lengths"].to(device))
        loss = compute_ctc_loss(output, batch)
        scaler.scale(loss / group_size).backward()
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip_norm"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        size = len(batch["texts"])
        total += loss.detach().item() * size
        count += size
    return {"loss": total / max(count, 1)}


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


@torch.inference_mode()
def validate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    total, samples, chars, words, char_errors, word_errors, exact, invalid = 0.0, 0, 0, 0, 0, 0, 0, 0
    for batch in loader:
        output = model(batch["mels"].to(device), batch["mel_lengths"].to(device))
        total += compute_ctc_loss(output, batch).item() * len(batch["texts"])
        for logits, length, reference in zip(output["logits"], output["lengths"], batch["texts"]):
            try:
                hypothesis = normalize_text(decode_ctc_path(logits[:int(length)].argmax(-1).tolist()))
            except UnicodeDecodeError:
                invalid += 1
                hypothesis = ""  # Count errors, never exclude failed decodes from denominator.
            samples += 1
            chars += len(reference)
            words += len(reference.split())
            char_errors += edit_distance(reference, hypothesis)
            word_errors += edit_distance(reference.split(), hypothesis.split())
            exact += hypothesis == reference
    if not samples:
        raise ValueError("Empty validation set")
    return {"loss": total / samples, "cer": char_errors / max(chars, 1),
            "wer_whitespace": word_errors / max(words, 1), "exact_match": exact / samples,
            "invalid_utf8_rate": invalid / samples, "samples": samples}


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, epoch=-1, metrics=None, best_cer=None):
    state = {"checkpoint_version": 1, "config": copy.deepcopy(model.config),
             "config_hash": config_hash(model.config), "model_state": model.state_dict(),
             "epoch": epoch, "metrics": metrics or {}, "best_cer": best_cer,
             "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    for key, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None:
            state[key] = obj.state_dict()
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


def fit(config, train_manifest, validation_manifest, output_dir, device="cpu", resume=None):
    validate_config(config)
    t = config["training"]
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    train, dev = ASRDataset(train_manifest, config), ASRDataset(validation_manifest, config)
    assert_disjoint(train, dev)
    audit, dev_audit = train.audit(), dev.audit()
    if audit["invalid_count"] or dev_audit["invalid_count"]:
        raise ValueError({"train_audit": audit, "validation_audit": dev_audit})
    sampler = LengthBucketSampler(audit["lengths"], t["batch_size"], t["seed"])
    if t["bucket_by_length"]:
        train_loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate_asr, num_workers=t["num_workers"])
    else:
        train_loader = DataLoader(train, batch_size=t["batch_size"], shuffle=True, collate_fn=collate_asr,
                                  num_workers=t["num_workers"])
    dev_loader = DataLoader(dev, batch_size=t["batch_size"], collate_fn=collate_asr, num_workers=t["num_workers"])
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] == "fp16")
    start, best = 0, float("inf")
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("config_hash") != config_hash(config) or saved["config"] != config:
            raise ValueError("Resume requires exactly compatible config")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start, best = saved["epoch"] + 1, saved["best_cer"]
    output_dir = Path(output_dir)
    save_config(config, output_dir / "config.json")
    for epoch in range(start, t["max_epochs"]):
        sampler.epoch = epoch
        training = train_one_epoch(model, train_loader, optimizer, scaler)
        metrics = validate(model, dev_loader)
        scheduler.step()
        improved = metrics["cer"] < best
        best = min(best, metrics["cer"])
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, metrics, best)
        if improved:
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, metrics, best)
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
    parser.add_argument("--resume")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else create_config()
    if args.write_default_config:
        save_config(config, args.write_default_config)
        return
    if not args.train_manifest:
        parser.error("--train-manifest is required")
    if args.audit_only:
        print(json.dumps(ASRDataset(args.train_manifest, config).audit(), ensure_ascii=False, indent=2))
        return
    if not args.validation_manifest or not args.output_dir:
        parser.error("--validation-manifest and --output-dir are required for training")
    fit(config, args.train_manifest, args.validation_manifest, args.output_dir, args.device, args.resume)


if __name__ == "__main__":
    main()
