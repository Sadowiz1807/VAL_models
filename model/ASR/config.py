"""Versioned configuration for the standalone Conformer/UTF-8-byte ASR.

This module deliberately has no torch dependency. JSON files are resources, not
Python modules. Defaults describe an untrained model, not a pretrained checkpoint.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def create_config() -> dict:
    return {
        "architecture": "conformer_byte_ctc_v1", "config_version": 1,
        "normalizer_version": "unicode_nfc_v1",
        "audio": {
            "sample_rate": 16000, "channels": 1, "frontend": "log_mel",
            "feature_dim": 80, "n_fft": 400, "win_length": 400,
            "hop_length": 160, "f_min": 0.0, "f_max": 8000.0,
            "center": False, "pad": 0, "power": 2.0, "norm": None,
            "mel_scale": "htk", "log_base": "natural", "log_floor": 1e-5,
            "cmvn": False,
        },
        "speech_encoder": {
            "d_model": 512, "layers": 8, "attention_heads": 8,
            "d_ff": 2048, "dropout": 0.1, "conv_kernel_size": 31,
            "subsampling_factor": 4, "subsampling_kernel": 3,
            "subsampling_padding": 1, "use_group_norm": True,
            "gradient_checkpointing": False,
        },
        "lexical_branch": {"type": "ctc_utf8_bytes", "vocab_size": 257, "blank_id": 0},
        "training": {
            "batch_size": 16, "gradient_accumulation_steps": 3,
            "learning_rate": 1e-4, "weight_decay": 0.01,
            "gradient_clip_norm": 1.0, "max_epochs": 100,
            "num_workers": 0, "amp": "off", "seed": 42,
            "scheduler": "cosine", "bucket_by_length": True,
            "allow_unknown_speakers": False, "metadata_audit": False,
            "valid_batch_size": 16, "pin_memory": False, "persistent_workers": False,
            "prefetch_factor": 2, "patience": 6, "min_delta": 0.0001,
        },
    }


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_config(config: dict) -> None:
    if config.get("architecture") != "conformer_byte_ctc_v1" or config.get("config_version") != 1:
        raise ValueError("Unsupported ASR architecture/config version; explicit migration required")
    if config.get("normalizer_version") != "unicode_nfc_v1":
        raise ValueError("Unsupported text normalizer")
    a, s, t = config["audio"], config["speech_encoder"], config["training"]
    for key in ("sample_rate", "feature_dim", "n_fft", "win_length", "hop_length"):
        _positive_int(a[key], f"audio.{key}")
    if a["channels"] != 1 or a["frontend"] != "log_mel":
        raise ValueError("Only mono log-Mel input is implemented")
    if a["center"] is not False or a["pad"] != 0 or a["win_length"] > a["n_fft"]:
        raise ValueError("Frontend requires center=False, pad=0, win_length<=n_fft")
    if not 0 <= a["f_min"] < a["f_max"] <= a["sample_rate"] / 2:
        raise ValueError("Invalid Mel frequency range")
    if (a["power"] != 2.0 or a["norm"] is not None or a["mel_scale"] != "htk"
            or a["log_base"] != "natural" or a["cmvn"] is not False):
        raise ValueError("Unsupported frontend profile; do not silently change preprocessing")
    if not math.isfinite(a["log_floor"]) or a["log_floor"] <= 0:
        raise ValueError("log_floor must be finite and positive")
    for key in ("d_model", "layers", "attention_heads", "d_ff", "conv_kernel_size"):
        _positive_int(s[key], f"speech_encoder.{key}")
    if s["d_model"] % s["attention_heads"] or s["conv_kernel_size"] % 2 != 1:
        raise ValueError("Invalid attention dimensions or even Conformer convolution kernel")
    if not 0 <= s["dropout"] < 1:
        raise ValueError("dropout must be in [0,1)")
    if s["subsampling_factor"] not in (2, 4):
        raise ValueError("Only 2x/4x subsampling is implemented")
    if s["subsampling_kernel"] != 3 or s["subsampling_padding"] != 1:
        raise ValueError("Subsampling implementation requires kernel=3, padding=1")
    if s["use_group_norm"] is not True or type(s["gradient_checkpointing"]) is not bool:
        raise ValueError("Invalid normalization/checkpointing profile")
    if config["lexical_branch"] != {"type": "ctc_utf8_bytes", "vocab_size": 257, "blank_id": 0}:
        raise ValueError("The byte codec requires 257 tokens and blank=0")
    for key in ("batch_size", "gradient_accumulation_steps", "max_epochs"):
        _positive_int(t[key], f"training.{key}")
    if type(t["num_workers"]) is not int or t["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    for key in ("learning_rate", "gradient_clip_norm"):
        if not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not math.isfinite(t["weight_decay"]) or t["weight_decay"] < 0:
        raise ValueError("Invalid weight_decay")
    if t["amp"] not in ("off", "fp16", "bf16") or t["scheduler"] != "cosine":
        raise ValueError("Unsupported AMP/scheduler setting")
    if type(t["seed"]) is not int or type(t["bucket_by_length"]) is not bool:
        raise ValueError("Invalid seed/bucketing setting")
    for key in ("allow_unknown_speakers", "metadata_audit", "pin_memory", "persistent_workers"):
        if key in t and type(t[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("valid_batch_size", "prefetch_factor", "patience"):
        if key in t:
            _positive_int(t[key], key)
    if "min_delta" in t and (not math.isfinite(t["min_delta"]) or t["min_delta"] < 0):
        raise ValueError("min_delta must be finite and nonnegative")


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def save_config(config: dict, path) -> None:
    validate_config(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return copy.deepcopy(config)
