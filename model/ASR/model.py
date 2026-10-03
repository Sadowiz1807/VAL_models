"""Audio frontend, strict byte codec, acoustic model and inference (no trainer imports)."""
from __future__ import annotations

import copy
import logging
import unicodedata

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchaudio.functional import resample
from torchaudio.models import Conformer
from torchaudio.transforms import MelSpectrogram

if __package__:
    from .config import create_config, validate_config, config_hash
else:  # Supports `python model/ASR/train.py` without an __init__.py.
    from config import create_config, validate_config, config_hash

LOGGER = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Transcript must be a string")
    return unicodedata.normalize("NFC", text)


def encode_utf8_target(text: str) -> list[int]:
    return [byte + 1 for byte in normalize_text(text).encode("utf-8")]


def min_ctc_frames(ids: list[int]) -> int:
    if any(type(token) is not int or not 1 <= token <= 256 for token in ids):
        raise ValueError("Target must contain non-blank byte IDs 1..256")
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))


def decode_ctc_path(path) -> str:
    kept, previous = [], None
    for token in path:
        token = int(token)
        if not 0 <= token <= 256:
            raise ValueError("CTC token outside byte vocabulary")
        if token != previous and token != 0:
            kept.append(token - 1)
        previous = token
    return bytes(kept).decode("utf-8", errors="strict")


def prepare_waveform(waveform, sample_rate: int, target_rate: int = 16000):
    """Input floats are PCM-scaled; integer arrays are rejected, not mis-scaled.

    Two-dimensional input must be [channels,samples]. File loading below handles
    soundfile's [samples,channels] layout explicitly.
    """
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    waveform = torch.as_tensor(waveform)
    if not waveform.is_floating_point():
        raise ValueError("Expected floating-point PCM waveform, not integer PCM")
    if waveform.ndim == 2:
        if not 1 <= waveform.size(0) <= 8:
            raise ValueError("Expected channels-first waveform")
        waveform = waveform.mean(0)
    if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError("Expected nonempty finite mono waveform")
    waveform = waveform.float()
    if sample_rate != target_rate:
        waveform = resample(waveform, sample_rate, target_rate)
    return waveform.contiguous()


def load_audio(path, target_rate: int = 16000):
    # soundfile avoids version-dependent torchaudio.load/torchcodec backends.
    import soundfile as sf
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return prepare_waveform(torch.from_numpy(audio.T.copy()), int(rate), target_rate), target_rate


def load_manifest_audio(row: dict, manifest_dir, target_rate: int = 16000):
    """Read WAV/FLAC or a bounded float32 segment from a prepared waveform shard.

    Offsets/counts are in samples, not bytes. Packed files are mono little-endian
    float32 PCM; workers read only their segment (no shared mutable file cursor).
    """
    from pathlib import Path
    if row.get("audio_format") != "f32le":
        return load_audio(Path(manifest_dir) / row["audio_ref"], target_rate)
    import numpy as np
    path = (Path(manifest_dir) / row["audio_ref"]).resolve()
    offset, count = row["audio_offset"], row["num_samples"]
    if type(offset) is not int or offset < 0 or type(count) is not int or count <= 0:
        raise ValueError("Invalid waveform shard offset/count")
    if row.get("channels") != 1 or row.get("sample_rate") != target_rate:
        raise ValueError("Packed waveform profile disagrees with audio config")
    if path.stat().st_size % 4 or (offset + count) * 4 > path.stat().st_size:
        raise ValueError("Truncated waveform shard")
    values = np.fromfile(path, dtype="<f4", count=count, offset=offset * 4)
    return prepare_waveform(torch.from_numpy(values), target_rate, target_rate), target_rate


def feature_length(sample_count: int, audio_config: dict) -> int:
    return max(0, (sample_count - audio_config["n_fft"]) // audio_config["hop_length"] + 1)


def encoder_length(length: int, factor: int = 4) -> int:
    if factor not in (2, 4):
        raise ValueError("Unsupported subsampling factor")
    return (length + factor - 1) // factor


def time_padding_mask(lengths, width):
    return torch.arange(width, device=lengths.device)[None, :] >= lengths[:, None]


class LogMelFrontend(nn.Module):
    def __init__(self, audio_config):
        super().__init__()
        self.config = copy.deepcopy(audio_config)
        a = audio_config
        self.mel = MelSpectrogram(
            sample_rate=a["sample_rate"], n_fft=a["n_fft"], win_length=a["win_length"],
            hop_length=a["hop_length"], f_min=a["f_min"], f_max=a["f_max"],
            n_mels=a["feature_dim"], center=a["center"], pad=a["pad"],
            power=a["power"], norm=a["norm"], mel_scale=a["mel_scale"],
        )

    def forward(self, waveform, sample_rate):
        if waveform.ndim != 1 or sample_rate != self.config["sample_rate"]:
            raise ValueError("Frontend requires mono waveform at configured sample rate")
        if waveform.numel() < self.config["n_fft"] or not torch.isfinite(waveform).all():
            raise ValueError("Audio too short or non-finite")
        return self.mel(waveform.float()).clamp_min(self.config["log_floor"]).log().transpose(0, 1)


class ByteCTCASR(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_config(config)
        self.config = copy.deepcopy(config)
        self.frontend = LogMelFrontend(config["audio"])
        s = config["speech_encoder"]
        d = s["d_model"]
        widths = [config["audio"]["feature_dim"]] + [d] * (1 if s["subsampling_factor"] == 2 else 2)
        self.subsample = nn.ModuleList([
            nn.Conv1d(a, b, kernel_size=3, stride=2, padding=1)
            for a, b in zip(widths, widths[1:])
        ])
        self.encoder = Conformer(
            input_dim=d, num_heads=s["attention_heads"], ffn_dim=s["d_ff"],
            num_layers=s["layers"], depthwise_conv_kernel_size=s["conv_kernel_size"],
            dropout=s["dropout"], use_group_norm=True,
        )
        self.ctc_head = nn.Linear(d, 257)

    def forward(self, mels, lengths):
        if mels.ndim != 3 or mels.size(-1) != self.config["audio"]["feature_dim"]:
            raise ValueError("Expected [B,T,feature_dim]")
        if lengths.shape != (mels.size(0),) or lengths.dtype != torch.long:
            raise ValueError("Expected int64 lengths [B]")
        lengths = lengths.to(mels.device)
        if (lengths <= 0).any() or (lengths > mels.size(1)).any() or not torch.isfinite(mels).all():
            raise ValueError("Invalid feature values/lengths")
        x = mels.masked_fill(time_padding_mask(lengths, mels.size(1))[..., None], 0).transpose(1, 2)
        for conv in self.subsample:
            x = torch.nn.functional.gelu(conv(x))
            lengths = (lengths + 1) // 2
            x = x.masked_fill(time_padding_mask(lengths, x.size(-1))[:, None, :], 0)
        x = x.transpose(1, 2)
        if self.training and self.config["speech_encoder"]["gradient_checkpointing"]:
            h, lengths = checkpoint(self.encoder, x, lengths, use_reentrant=False)
        else:
            h, lengths = self.encoder(x, lengths)
        return {"logits": self.ctc_head(h), "lengths": lengths}

    @torch.inference_mode()
    def transcribe(self, waveform, sample_rate: int, request_id=None) -> dict:
        """No VAD is implied: empty CTC output is not a proven no-speech decision."""
        self.eval()
        device = next(self.parameters()).device
        base = {"request_id": request_id, "asr_version": self.config["architecture"]}
        try:
            waveform = prepare_waveform(waveform, sample_rate, self.config["audio"]["sample_rate"])
            if waveform.numel() < self.config["audio"]["n_fft"]:
                return {**base, "status": "audio_too_short", "text": None}
            features = self.frontend(waveform.to(device), self.config["audio"]["sample_rate"])
        except (ValueError, TypeError) as error:
            return {**base, "status": "invalid_audio", "text": None, "error": str(error)}
        output = self(features[None], torch.tensor([len(features)], device=device, dtype=torch.long))
        logits = output["logits"][0, :int(output["lengths"][0])]
        if not torch.isfinite(logits).all():
            return {**base, "status": "nonfinite_output", "text": None}
        try:
            text = normalize_text(decode_ctc_path(logits.argmax(-1).tolist()))
        except UnicodeDecodeError:
            LOGGER.warning("Invalid UTF-8 CTC path for request %s", request_id)
            return {**base, "status": "invalid_utf8", "text": None}
        return {**base, "status": "ok" if text.strip() else "empty_transcript", "text": text}


def build_model(config=None) -> ByteCTCASR:
    return ByteCTCASR(create_config() if config is None else config)


def load_for_inference(checkpoint_path, device="cpu") -> ByteCTCASR:
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved.get("checkpoint_version") != 1:
        raise ValueError("Unsupported ASR checkpoint")
    config = saved["config"]
    if saved.get("config_hash") != config_hash(config):
        raise ValueError("Checkpoint config hash mismatch")
    model = build_model(config)
    model.load_state_dict(saved["model_state"], strict=True)
    return model.to(device).eval()


def transcribe(model: ByteCTCASR, waveform, sample_rate: int, request_id=None):
    return model.transcribe(waveform, sample_rate, request_id)
