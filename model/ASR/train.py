"""ASR JSONL dataset, CTC training, validation and checkpoint CLI.

No training happens on import. Run `python model/ASR/train.py --help`.
Manifests require dataset_id, sample_id, audio_ref, transcript. Speaker/session
IDs may be null only with explicit training.allow_unknown_speakers=True.
audio_ref is relative to the manifest directory; f32le shards also require sample
offset/count/rate. --prepare-bud500 prepares local Parquet without training.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import tempfile
import hashlib
import io
import re
import shutil
import sqlite3
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

if __package__:
    from .config import create_config, load_config, save_config, validate_config, config_hash
    from .model import (build_model, LogMelFrontend, load_audio, load_manifest_audio, encode_utf8_target,
                        min_ctc_frames, decode_ctc_path, normalize_text, encoder_length, feature_length,
                        prepare_waveform)
else:
    from config import create_config, load_config, save_config, validate_config, config_hash
    from model import (build_model, LogMelFrontend, load_audio, load_manifest_audio, encode_utf8_target,
                       min_ctc_frames, decode_ctc_path, normalize_text, encoder_length, feature_length,
                       prepare_waveform)


class ManifestRows(Sequence):
    """Offset-indexed JSONL: do not pickle 600k dictionaries into every Windows worker."""
    def __init__(self, path):
        import numpy as np
        self.path = Path(path).resolve()
        offsets = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        self.offsets = np.asarray(offsets, dtype=np.int64)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        with self.path.open("rb") as stream:
            stream.seek(int(self.offsets[index]))
            return json.loads(stream.readline())

    def __iter__(self):
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


class ASRDataset(Dataset):
    def __init__(self, manifest, config):
        validate_config(config)
        self.path, self.config = Path(manifest), copy.deepcopy(config)
        self.frontend = LogMelFrontend(config["audio"])
        self.rows = ManifestRows(self.path)
        if not self.rows:
            raise ValueError("Empty ASR manifest")
        seen = set()
        unknown = 0
        for row in self.rows:
            for field in ("dataset_id", "sample_id", "audio_ref", "transcript"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Missing/invalid {field}: {row.get('sample_id')}")
            for field in ("speaker_id", "session_id"):
                if row.get(field) is None and config["training"].get("allow_unknown_speakers", False):
                    unknown += 1
                elif not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Missing/invalid {field}: explicit allow_unknown_speakers required")
            key = (row["dataset_id"], row["sample_id"])
            if key in seen:
                raise ValueError(f"Duplicate sample {key}")
            seen.add(key)
        if unknown:
            warnings.warn("Speaker/session metadata unavailable: speaker-disjointness is NOT verified", stacklevel=2)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio, rate = load_manifest_audio(row, self.path.parent, self.config["audio"]["sample_rate"])
        with torch.no_grad():
            features = self.frontend(audio, rate)
        targets = encode_utf8_target(row["transcript"])
        return {"features": features, "targets": targets,
                "sample_id": row["sample_id"], "text": normalize_text(row["transcript"])}

    def audit(self, metadata_only=False):
        """Full audio audit by default; fast mode requires prepared metadata.

        Fast mode checks shard ranges, transcript lengths and the exact frontend
        profile. It does NOT reread every audio sample or revalidate its hash.
        """
        lengths, errors = [], []
        sizes = {}
        factor = self.config["speech_encoder"]["subsampling_factor"]
        for index, row in enumerate(self.rows):
            try:
                if metadata_only:
                    if row.get("preparation_version") != 1 or row.get("audio_format") != "f32le":
                        raise ValueError("Metadata audit only supports prepared f32le manifests")
                    a = self.config["audio"]
                    if row["sample_rate"] != a["sample_rate"] or row["channels"] != 1:
                        raise ValueError("Prepared audio profile mismatch")
                    reference = row["audio_ref"]
                    if reference not in sizes:
                        sizes[reference] = (self.path.parent / reference).resolve().stat().st_size
                    offset, count = row["audio_offset"], row["num_samples"]
                    if type(offset) is not int or offset < 0 or type(count) is not int or count < a["n_fft"]:
                        raise ValueError("Invalid packed waveform range")
                    if (offset + count) * 4 > sizes[reference] or sizes[reference] % 4:
                        raise ValueError("Truncated waveform shard")
                    length = feature_length(count, a)
                    required = min_ctc_frames(encode_utf8_target(row["transcript"]))
                    if required != row["min_ctc_frames"]:
                        raise ValueError("Transcript/metadata mismatch")
                else:
                    item = self[index]
                    length = len(item["features"])
                    required = min_ctc_frames(item["targets"])
                lengths.append(length)
                if encoder_length(length, factor) < required:
                    errors.append(f"{row['sample_id']}: infeasible byte CTC alignment")
            except (ValueError, RuntimeError, OSError) as error:
                lengths.append(0)
                errors.append(f"{row['sample_id']}: {error}")
        return {"samples": len(self), "invalid_count": len(errors),
                "invalid_rate": len(errors) / len(self), "errors": errors, "lengths": lengths,
                "mode": "metadata" if metadata_only else "full_audio"}


def assert_disjoint(train: ASRDataset, validation: ASRDataset):
    for field in ("sample_id", "speaker_id", "session_id", "waveform_sha256"):
        left = {(r["dataset_id"], r[field]) for r in train.rows if r.get(field) is not None}
        right = {(r["dataset_id"], r[field]) for r in validation.rows if r.get(field) is not None}
        if left & right:
            raise ValueError(f"Train/validation leakage in {field}: {list(left & right)[:3]}")
    def audio_segments(dataset):
        resolved = {}
        segments = set()
        for row in dataset.rows:
            reference = row["audio_ref"]
            if reference not in resolved:
                resolved[reference] = (dataset.path.parent / reference).resolve()
            segments.add((resolved[reference], row.get("audio_offset"), row.get("num_samples")))
        return segments
    left_audio = audio_segments(train)
    right_audio = audio_segments(validation)
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
    def __init__(self, lengths, batch_size, seed, shuffle=True):
        self.lengths, self.batch_size, self.seed, self.epoch = lengths, batch_size, seed, 0
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        else:
            indices.sort(key=lambda i: self.lengths[i])
        batches = []
        window = self.batch_size * 50
        for start in range(0, len(indices), window):
            bucket = sorted(indices[start:start + window], key=lambda i: self.lengths[i])
            batches.extend(bucket[i:i + self.batch_size] for i in range(0, len(bucket), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)


def _amp(config, device):
    setting = config["training"]["amp"]
    if setting == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training requires CUDA")
    return torch.autocast(device_type=device.type, enabled=setting != "off",
                          dtype=torch.float16 if setting == "fp16" else torch.bfloat16)


def _progress(loader, description):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return loader
    return tqdm(loader, total=len(loader), desc=description, unit="batch", leave=False)


def train_one_epoch(model, loader, optimizer, scaler=None):
    device = next(model.parameters()).device
    t = model.config["training"]
    scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, count = 0.0, 0
    accumulation = t["gradient_accumulation_steps"]
    for index, batch in enumerate(_progress(loader, "ASR train")):
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
    for batch in _progress(loader, "ASR validation"):
        with _amp(model.config, device):
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


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, epoch=-1, metrics=None, best_cer=None,
                    run_state=None):
    state = {"checkpoint_version": 1, "config": copy.deepcopy(model.config),
             "config_hash": config_hash(model.config), "model_state": model.state_dict(),
             "epoch": epoch, "metrics": metrics or {}, "best_cer": best_cer,
             "run_state": run_state or {},
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


def build_loaders(config, train_manifest, validation_manifest):
    """Module-level dataset/collate/worker init supports Windows notebook spawn."""
    t = config["training"]
    train, dev = ASRDataset(train_manifest, config), ASRDataset(validation_manifest, config)
    assert_disjoint(train, dev)
    audit = train.audit(metadata_only=t.get("metadata_audit", False))
    dev_audit = dev.audit(metadata_only=t.get("metadata_audit", False))
    if audit["invalid_count"] or dev_audit["invalid_count"]:
        raise ValueError({"train_errors": audit["errors"][:20], "validation_errors": dev_audit["errors"][:20]})
    workers = t["num_workers"]
    kwargs = {"collate_fn": collate_asr, "num_workers": workers, "pin_memory": t.get("pin_memory", False)}
    if workers:
        kwargs.update(persistent_workers=t.get("persistent_workers", False),
                      prefetch_factor=t.get("prefetch_factor", 2), worker_init_fn=initialize_worker)
    if t["bucket_by_length"]:
        train_sampler = LengthBucketSampler(audit["lengths"], t["batch_size"], t["seed"])
        train_loader = DataLoader(train, batch_sampler=train_sampler, **kwargs)
    else:
        train_loader = DataLoader(train, batch_size=t["batch_size"], shuffle=True, **kwargs)
    dev_sampler = LengthBucketSampler(dev_audit["lengths"], t.get("valid_batch_size", t["batch_size"]), t["seed"], shuffle=False)
    dev_loader = DataLoader(dev, batch_sampler=dev_sampler, **kwargs)
    return train_loader, dev_loader


def initialize_worker(worker_id):
    import numpy as np
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def fit(config, train_manifest, validation_manifest, output_dir, device="cpu", resume=None):
    validate_config(config)
    t = config["training"]
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    train_loader, dev_loader = build_loaders(config, train_manifest, validation_manifest)
    manifest_hashes = {"train": _sha256_file(train_manifest), "validation": _sha256_file(validation_manifest)}
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] == "fp16")
    start, best, bad_epochs = 0, float("inf"), 0
    early_stop_best = float("inf")
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("config_hash") != config_hash(config) or saved["config"] != config:
            raise ValueError("Resume requires exactly compatible config")
        if saved.get("run_state", {}).get("manifest_hashes") != manifest_hashes:
            raise ValueError("Resume requires identical train/validation manifests")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start, best = saved["epoch"] + 1, saved["best_cer"]
        bad_epochs = saved["run_state"].get("bad_epochs", 0)
        early_stop_best = saved["run_state"].get("early_stop_best", best)
    output_dir = Path(output_dir)
    if not resume and any((output_dir / name).exists() for name in ("last.pt", "best.pt", "metrics.jsonl")):
        raise FileExistsError("Existing training run: choose resume or a new output directory")
    save_config(config, output_dir / "config.json")
    for epoch in range(start, t["max_epochs"]):
        if hasattr(train_loader.batch_sampler, "epoch"):
            train_loader.batch_sampler.epoch = epoch
        training = train_one_epoch(model, train_loader, optimizer, scaler)
        metrics = validate(model, dev_loader)
        scheduler.step()
        improved = metrics["cer"] < best
        best = min(best, metrics["cer"])
        if metrics["cer"] < early_stop_best - t.get("min_delta", 0.0001):
            early_stop_best, bad_epochs = metrics["cer"], 0
        else:
            bad_epochs += 1
        run_state = {"manifest_hashes": manifest_hashes, "bad_epochs": bad_epochs, "early_stop_best": early_stop_best}
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, metrics, best, run_state)
        if improved:
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, metrics, best, run_state)
        record = {"epoch": epoch, "train": training, "validation": metrics}
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if bad_epochs >= t.get("patience", 6):
            print(f"Early stopping: {bad_epochs} epochs without CER improvement", flush=True)
            break
    return model


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _prepare_bud_shard(source, output_root):
    """One bounded Arrow batch at a time; original Parquet is never modified."""
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    source, output_root = Path(source), Path(output_root)
    split, shard_number = source.name.split("-")[:2]
    name = f"{split}-{shard_number}"
    target = output_root / "shards" / name
    signature = {"name": source.name, "size": source.stat().st_size,
                 "mtime_ns": source.stat().st_mtime_ns, "preparation_version": 1}
    if (target / "done.json").exists():
        with (target / "done.json").open(encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved["source"] != signature:
            raise ValueError(f"Source changed since preparation: {source}")
        for file, size in saved["files"].items():
            if not (target / file).is_file() or (target / file).stat().st_size != size:
                raise ValueError(f"Incomplete/corrupt prepared shard: {target / file}")
        return saved
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)  # Only our uncommitted output; never source data.
    partial.mkdir(parents=True)
    records_path, reject_path = partial / "records.jsonl", partial / "rejected.jsonl"
    counts, sample_rates, channel_counts = Counter(), Counter(), Counter()
    total_samples = offset = 0
    minimum, maximum = None, 0
    parquet = pq.ParquetFile(source)
    if parquet.schema_arrow.names != ["audio", "transcription"]:
        raise ValueError(f"Unexpected Bud500 schema: {parquet.schema_arrow}")
    with (partial / "audio.f32").open("wb") as audio_out, records_path.open("w", encoding="utf-8") as records, reject_path.open("w", encoding="utf-8") as rejected:
        row_index = 0
        for batch in parquet.iter_batches(batch_size=64, columns=["audio", "transcription"], use_threads=False):
            for row in batch.to_pylist():
                index = row_index
                row_index += 1
                sample_id = f"viet_bud500:{split}:{shard_number}:{index:06d}"
                counts["source_rows"] += 1
                try:
                    raw = row["transcription"]
                    if not isinstance(raw, str) or not raw.strip():
                        raise ValueError("Missing/empty source transcript")
                    # Preserve spelling, punctuation/case and all whitespace. Only NFC.
                    text = normalize_text(raw)
                    counts["nfc_changed"] += text != raw
                    payload = row["audio"].get("bytes")
                    if not payload:
                        raise ValueError("Missing embedded audio bytes; path fallback is not guessed")
                    values, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
                    channels = values.shape[1]
                    sample_rates[str(rate)] += 1
                    channel_counts[str(channels)] += 1
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite source audio")
                    # PCM scaling from libsndfile; no per-utterance peak normalization,
                    # silence trimming, clipping or time stretching.
                    waveform = values.mean(axis=1, dtype=np.float32)
                    if rate != 16000:
                        waveform = prepare_waveform(torch.from_numpy(waveform), int(rate), 16000).numpy()
                    waveform = np.asarray(waveform, dtype="<f4")
                    if len(waveform) < 400:
                        raise ValueError("Audio shorter than one frontend window")
                    peak = float(np.abs(waveform).max())
                    if not math.isfinite(peak) or peak > 1.0001:
                        raise ValueError("Waveform outside normalized PCM range")
                    if peak == 0:
                        raise ValueError("All-zero waveform with nonempty transcript")
                    samples = len(waveform)
                    mels = (samples - 400) // 160 + 1
                    required = min_ctc_frames(encode_utf8_target(text))
                    packed = waveform.tobytes()
                    waveform_hash = hashlib.sha256(packed).hexdigest()
                    record = {
                        "preparation_version": 1, "dataset_id": "viet_bud500", "sample_id": sample_id,
                        "split": split, "audio_ref": f"shards/{name}/audio.f32", "audio_format": "f32le",
                        "audio_offset": offset, "num_samples": samples, "sample_rate": 16000, "channels": 1,
                        "duration_seconds": samples / 16000, "transcript": text,
                        "transcript_source": "parquet.transcription", "normalizer_version": "unicode_nfc_v1",
                        "speaker_id": None, "session_id": None, "speaker_metadata_status": "not_provided_by_source",
                        "original_audio_id": waveform_hash, "waveform_sha256": waveform_hash,
                        "source_shard": source.name, "source_row": index, "source_audio_path": row["audio"].get("path"),
                        "mel_frames": mels, "target_bytes": len(text.encode("utf-8")), "min_ctc_frames": required,
                        "ctc_feasible_2x": encoder_length(mels, 2) >= required,
                        "ctc_feasible_4x": encoder_length(mels, 4) >= required,
                    }
                    audio_out.write(packed)
                    records.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    offset += samples
                    total_samples += samples
                    counts["prepared"] += 1
                    counts["infeasible_2x"] += not record["ctc_feasible_2x"]
                    counts["infeasible_4x"] += not record["ctc_feasible_4x"]
                    minimum = samples if minimum is None else min(minimum, samples)
                    maximum = max(maximum, samples)
                except (ValueError, TypeError, RuntimeError, KeyError) as error:
                    counts["invalid"] += 1
                    rejected.write(json.dumps({"sample_id": sample_id, "split": split, "source_shard": source.name,
                                               "source_row": index, "reason": str(error)}, ensure_ascii=False) + "\n")
    result = {"source": signature, "counts": dict(counts), "sample_rates": dict(sample_rates),
              "channel_counts": dict(channel_counts), "total_samples": total_samples,
              "min_samples": minimum, "max_samples": maximum,
              "files": {p.name: p.stat().st_size for p in partial.iterdir()},
              "source_sha256": _sha256_file(source),
              "audio_sha256": _sha256_file(partial / "audio.f32")}
    _json_dump(partial / "done.json", result)
    os.replace(partial, target)
    return result


def prepare_bud500(source_dir, output_dir, workers=4):
    """Export every source utterance to float32 shards and versioned manifests.

    Official splits are retained. Exact normalized-audio duplicates crossing splits
    are quarantined from ALL training/evaluation manifests (master retains them).
    Speaker/session absence is explicit, never replaced with invented identities.
    Resume is per committed shard; a final preparation lock prevents two writers.
    """
    import pyarrow.parquet as pq
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Prepared output must be outside the raw dataset directory")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    sources = sorted((source_dir / "data").glob("*.parquet"))
    if not sources:
        raise ValueError("No local Parquet shards found")
    groups = {}
    expected_rows = Counter()
    for path in sources:
        match = re.fullmatch(r"(train|validation|test)-(\d+)-of-(\d+)-.+\.parquet", path.name)
        if not match:
            raise ValueError(f"Unexpected shard name: {path.name}")
        split, index, total = match.group(1), int(match.group(2)), int(match.group(3))
        groups.setdefault(split, []).append((index, total))
        expected_rows[split] += pq.ParquetFile(path).metadata.num_rows
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("Need complete official train/validation/test splits")
    for split, shards in groups.items():
        total = shards[0][1]
        if any(n != total for _, n in shards) or sorted(i for i, _ in shards) != list(range(total)):
            raise ValueError(f"Missing/duplicate {split} shards")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".preparation.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, f"pid={os.getpid()}\n".encode())
    os.close(descriptor)
    started = time.time()
    try:
        # A completed report is the commit marker; stale outputs must not train.
        report_path = output_dir / "preparation_report.json"
        if report_path.exists():
            report_path.unlink()
        reserve = 10 * 2**30
        pending = sum(p.stat().st_size for p in sources
                      if not (output_dir / "shards" / "-".join(p.name.split("-")[:2]) / "done.json").exists())
        if shutil.disk_usage(output_dir).free < pending * 2.2 + reserve:
            raise OSError("Insufficient free disk for float32 waveforms plus safety reserve")
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_prepare_bud_shard, p, output_dir): p for p in sources}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({"shards_done": len(results), "shards_total": len(sources),
                                  "shard": futures[future].name, "counts": result["counts"],
                                  "elapsed_seconds": round(time.time() - started, 1)}), flush=True)
        # SQLite keeps exact cross-split duplicate detection bounded in RAM.
        db_path = output_dir / ".duplicate-index.sqlite"
        if db_path.exists():
            db_path.unlink()
        database = sqlite3.connect(db_path)
        try:
            database.execute("CREATE TABLE audio (hash TEXT PRIMARY KEY, split TEXT, first_id TEXT, cross_split INTEGER DEFAULT 0)")
            for source in sources:
                folder = output_dir / "shards" / "-".join(source.name.split("-")[:2])
                with (folder / "records.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        database.execute("INSERT INTO audio(hash,split,first_id) VALUES(?,?,?) "
                                         "ON CONFLICT(hash) DO UPDATE SET cross_split=MAX(audio.cross_split, audio.split != excluded.split)",
                                         (row["waveform_sha256"], row["split"], row["sample_id"]))
                database.commit()
            report = {
                "preparation_version": 1, "status": "complete", "dataset_id": "viet_bud500",
                "source_dir": str(source_dir), "source_rows": dict(expected_rows), "source_shards": len(sources),
                "waveform_format": "mono little-endian float32 PCM, 16000 Hz", "transcript_source": "original parquet.transcription",
                "normalization": "Unicode NFC only; no relabeling or punctuation/case/whitespace stripping",
                "audio_normalization": "PCM to float32; mono/resample if required; no peak normalization, trimming or clipping",
                "speaker_session_disjointness": "UNVERIFIED: source supplies no speaker/session IDs; user accepted official splits",
                "duplicate_policy": "Quarantine every occurrence of exact waveform hashes shared across splits; retain within-split duplicates",
                "license_warning": "Source README metadata says CC-BY-NC-SA-4.0 while license section/file says Apache-2.0; resolve before commercial use",
                "counts": {}, "manifests": {}, "shards": sorted(results, key=lambda x: x["source"]["name"]),
            }
            counters = {split: Counter() for split in groups}
            handles = {}
            names = ["manifest.jsonl", "transcripts.jsonl", "quarantine.jsonl"]
            names += [f"{split}.ctc{factor}.jsonl" for split in groups for factor in (2, 4)]
            try:
                handles = {name: (output_dir / (name + ".tmp")).open("w", encoding="utf-8") for name in names}
                for source in sources:
                    folder = output_dir / "shards" / "-".join(source.name.split("-")[:2])
                    with (folder / "rejected.jsonl").open(encoding="utf-8") as stream:
                        for line in stream:
                            handles["quarantine.jsonl"].write(line)
                            counters[json.loads(line)["split"]]["invalid_audio_or_transcript"] += 1
                    with (folder / "records.jsonl").open(encoding="utf-8") as stream:
                        for line in stream:
                            row = json.loads(line)
                            cross, first = database.execute("SELECT cross_split,first_id FROM audio WHERE hash=?", (row["waveform_sha256"],)).fetchone()
                            row["cross_split_audio_duplicate"] = bool(cross)
                            row["duplicate_of"] = first if first != row["sample_id"] else None
                            rendered = json.dumps(row, ensure_ascii=False) + "\n"
                            handles["manifest.jsonl"].write(rendered)
                            handles["transcripts.jsonl"].write(json.dumps({"sample_id": row["sample_id"], "split": row["split"],
                                "transcript": row["transcript"], "source_shard": row["source_shard"], "source_row": row["source_row"]}, ensure_ascii=False) + "\n")
                            counts = counters[row["split"]]
                            counts["prepared"] += 1
                            counts["samples"] += row["num_samples"]
                            counts["cross_split_duplicate_rows"] += bool(cross)
                            counts["within_split_duplicate_rows"] += not cross and row["duplicate_of"] is not None
                            for factor in (2, 4):
                                feasible = row[f"ctc_feasible_{factor}x"]
                                counts[f"infeasible_ctc{factor}"] += not feasible
                                if feasible and not cross:
                                    handles[f"{row['split']}.ctc{factor}.jsonl"].write(rendered)
                                    counts[f"eligible_ctc{factor}"] += 1
                            reasons = []
                            if cross:
                                reasons.append("exact_audio_duplicate_across_official_splits")
                            if not row["ctc_feasible_2x"]:
                                reasons.append("infeasible_byte_ctc_even_at_2x")
                            if reasons:
                                handles["quarantine.jsonl"].write(json.dumps({"sample_id": row["sample_id"], "split": row["split"], "reasons": reasons}, ensure_ascii=False) + "\n")
            finally:
                for handle in handles.values():
                    handle.close()
            for name in names:
                os.replace(output_dir / (name + ".tmp"), output_dir / name)
                report["manifests"][name] = {"bytes": (output_dir / name).stat().st_size, "sha256": _sha256_file(output_dir / name)}
            report["counts"] = {split: dict(counts) for split, counts in counters.items()}
            for split, counts in counters.items():
                if counts["prepared"] + counts["invalid_audio_or_transcript"] != expected_rows[split]:
                    raise ValueError("Preparation row count mismatch")
            report["total_hours"] = sum(x["samples"] for x in counters.values()) / 16000 / 3600
            report["elapsed_seconds"] = round(time.time() - started, 2)
            _json_dump(report_path, report)
            return report
        finally:
            database.close()
    finally:
        lock.unlink(missing_ok=True)


def resplit_bud500(prepared_dir, output_dir, test_fraction=0.20, random_state=42):
    """Reassign eligible CTC-2x train+test, retaining original validation membership.

    Shuffle sorted exact-waveform hash groups with NumPy RandomState/MT19937.
    Greedily take whole groups until ceil(fraction * ALL eligible samples) are
    selected; fail if that exact count is not representable without splitting a
    group. This is group-aware random splitting, not sklearn row-wise splitting.
    Waveforms and original manifests are read-only. Commit a NEW directory only.
    """
    import numpy as np
    prepared_dir, output_dir = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    if type(random_state) is not int or not 0 <= random_state < 2**32:
        raise ValueError("random_state must be a uint32 integer")
    if type(test_fraction) not in (int, float) or not math.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be finite and in (0,1)")
    if output_dir == prepared_dir or output_dir in prepared_dir.parents:
        raise ValueError("Use a new split subdirectory, not the prepared/source root")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing split: {output_dir}")
    if (prepared_dir / ".preparation.lock").exists():
        raise RuntimeError("Preparation is active or has an unreviewed stale lock")
    report_path = prepared_dir / "preparation_report.json"
    preparation_hash = _sha256_file(report_path)
    with report_path.open(encoding="utf-8") as stream:
        preparation = json.load(stream)
    if preparation.get("status") != "complete" or preparation.get("preparation_version") != 1:
        raise ValueError("Requires a complete version-1 prepared dataset")
    source_paths = {split: prepared_dir / f"{split}.ctc2.jsonl" for split in ("train", "validation", "test")}
    source_hashes = {}
    for split, path in source_paths.items():
        source_hashes[split] = _sha256_file(path)
        if source_hashes[split] != preparation["manifests"][path.name]["sha256"]:
            raise ValueError(f"Source manifest hash mismatch: {path}")

    group_sizes, validation_groups, seen_ids = Counter(), set(), set()
    source_counts = Counter()
    for split, path in source_paths.items():
        for row in ManifestRows(path):
            key = (row["dataset_id"], row["sample_id"])
            if key in seen_ids:
                raise ValueError(f"Duplicate sample ID across source manifests: {key}")
            seen_ids.add(key)
            if row["split"] != split or not row.get("ctc_feasible_2x") or row.get("cross_split_audio_duplicate"):
                raise ValueError(f"Ineligible/misassigned source row: {key}")
            waveform_hash = row.get("waveform_sha256")
            if not isinstance(waveform_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", waveform_hash):
                raise ValueError(f"Missing/invalid waveform SHA256: {key}")
            source_counts[split] += 1
            if split == "validation":
                validation_groups.add(waveform_hash)
            else:
                group_sizes[waveform_hash] += 1
        if source_counts[split] != preparation["counts"][split]["eligible_ctc2"]:
            raise ValueError(f"Eligible source row count mismatch: {split}")
    if validation_groups.intersection(group_sizes):
        raise ValueError("An audio group crosses fixed validation and train/test; cannot retain validation safely")
    total = sum(source_counts.values())
    requested_test = math.ceil(total * test_fraction)
    if not 0 < requested_test < source_counts["train"] + source_counts["test"]:
        raise ValueError("Test fraction leaves no training data after retaining validation")
    del seen_ids
    groups = sorted(group_sizes)
    order = np.random.RandomState(random_state).permutation(len(groups))
    test_groups, remaining = set(), requested_test
    for index in order:
        group = groups[int(index)]
        size = group_sizes[group]
        if size <= remaining:
            test_groups.add(group)
            remaining -= size
        if not remaining:
            break
    if remaining:
        raise ValueError("Exact test sample count cannot be met while retaining whole duplicate groups")
    expected = {"train": total - source_counts["validation"] - requested_test,
                "validation": source_counts["validation"], "test": requested_test}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".partial-", dir=output_dir.parent))
    try:
        from contextlib import ExitStack
        counts, total_samples, origin_counts = Counter(), Counter(), {split: Counter() for split in expected}
        rebased_paths = {}
        with ExitStack() as stack:
            streams = {split: stack.enter_context((staging / f"{split}.ctc2.jsonl").open("w", encoding="utf-8"))
                       for split in expected}
            for source_split, path in source_paths.items():
                for row in ManifestRows(path):
                    split = "validation" if source_split == "validation" else (
                        "test" if row["waveform_sha256"] in test_groups else "train")
                    reference = row["audio_ref"]
                    if reference not in rebased_paths:
                        waveform_path = (prepared_dir / reference).resolve()
                        if not waveform_path.is_file():
                            raise FileNotFoundError(waveform_path)
                        rebased_paths[reference] = Path(os.path.relpath(waveform_path, output_dir)).as_posix()
                    row["audio_ref"] = rebased_paths[reference]
                    row["original_split"] = source_split
                    row["split"] = split
                    row["split_version"] = "bud500_group_random_ctc2_v1"
                    streams[split].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    counts[split] += 1
                    total_samples[split] += row["num_samples"]
                    origin_counts[split][source_split] += 1
        if dict(counts) != expected:
            raise ValueError(f"Unexpected output counts: {dict(counts)} != {expected}")
        if (prepared_dir / ".preparation.lock").exists() or _sha256_file(report_path) != preparation_hash:
            raise RuntimeError("Prepared dataset changed during split creation")
        for split, path in source_paths.items():
            if _sha256_file(path) != source_hashes[split]:
                raise RuntimeError(f"Source changed during split creation: {path}")
        manifests = {f"{split}.ctc2.jsonl": {"bytes": (staging / f"{split}.ctc2.jsonl").stat().st_size,
                                              "sha256": _sha256_file(staging / f"{split}.ctc2.jsonl")}
                     for split in expected}
        report = {
            "status": "complete", "split_version": "bud500_group_random_ctc2_v1", "random_state": random_state,
            "algorithm": "NumPy RandomState(MT19937); permute sorted waveform SHA256 groups; exact-count whole-group greedy selection",
            "requested_test_fraction": test_fraction, "actual_test_fraction": counts["test"] / total,
            "ratio_denominator": "all eligible CTC-2x samples, including unchanged validation",
            "total_eligible_samples": total, "raw_source_samples": sum(preparation["source_rows"].values()),
            "excluded_before_split": sum(preparation["source_rows"].values()) - total,
            "source_preparation_report_sha256": preparation_hash, "source_manifest_sha256": source_hashes,
            "source_eligible_counts": dict(source_counts), "counts": dict(counts),
            "hours": {split: total_samples[split] / 16000 / 3600 for split in expected},
            "original_split_counts": {split: dict(value) for split, value in origin_counts.items()},
            "validation_policy": "Preserve every original eligible validation sample; never move any into train/test",
            "group_key": "waveform_sha256", "nonvalidation_audio_groups": len(groups),
            "duplicate_extra_rows_grouped": sum(size - 1 for size in group_sizes.values()),
            "speaker_session_disjointness": "UNVERIFIED: no speaker/session metadata; only exact waveform groups are isolated",
            "manifest_order": "stable source order; random membership, DataLoader shuffles train batches",
            "manifests": manifests,
            "checkpoint_policy": "New training run required; do not evaluate old-split trained weights on the new test set",
        }
        _json_dump(staging / "split_report.json", report)
        os.rename(staging, output_dir)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


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
    parser.add_argument("--prepare-bud500", help="Local raw dataset directory; no download")
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument("--resplit-bud500", help="Prepared dataset root; retain validation and randomly repartition eligible train+test")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if args.prepare_bud500 and args.resplit_bud500:
        parser.error("Preparation and resplitting are mutually exclusive")
    if args.resplit_bud500:
        if not args.output_dir:
            parser.error("Resplitting requires a NEW --output-dir")
        report = resplit_bud500(args.resplit_bud500, args.output_dir, args.test_fraction, args.random_state)
        print(json.dumps({"status": report["status"], "counts": report["counts"], "test_fraction": report["actual_test_fraction"]}))
        return
    if args.prepare_bud500:
        if not args.output_dir:
            parser.error("Preparation requires --output-dir")
        report = prepare_bud500(args.prepare_bud500, args.output_dir, args.prepare_workers)
        print(json.dumps({"status": report["status"], "counts": report["counts"], "hours": report["total_hours"]}))
        return
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
