"""Audio frontend, strict byte codec, acoustic model and inference (no trainer imports)."""
from __future__ import annotations

import copy
import logging
import math
import time
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


def ctc_path_to_bytes(path) -> bytes:
    """Collapse repeated frame tokens FIRST, then discard blank=0 (not byte NUL)."""
    kept, previous = [], None
    for token in path:
        token = int(token)
        if not 0 <= token <= 256:
            raise ValueError("CTC token outside byte vocabulary")
        if token != previous and token != 0:
            kept.append(token - 1)
        previous = token
    return bytes(kept)


def decode_ctc_path(path) -> str:
    """Strict legacy decoder, also used by validation; never repairs bytes."""
    return ctc_path_to_bytes(path).decode("utf-8", errors="strict")


def utf8_next_state(state: int, byte: int):
    """RFC 3629 incremental validator; None rejects, 0 is a complete code point.

    States 1/2/3 expect ordinary continuation bytes; 4/5/6/7 encode the special
    first-continuation bounds for E0/ED/F0/F4. Update ONLY on emitted CTC bytes.
    """
    if type(state) is not int or not 0 <= state <= 7 or type(byte) is not int or not 0 <= byte <= 255:
        raise ValueError("Invalid UTF-8 state/byte")
    if state == 0:
        if byte <= 0x7f:
            return 0
        if 0xc2 <= byte <= 0xdf:
            return 1
        if byte == 0xe0:
            return 4
        if byte == 0xed:
            return 5
        if 0xe1 <= byte <= 0xef:
            return 2
        if byte == 0xf0:
            return 6
        if byte == 0xf4:
            return 7
        if 0xf1 <= byte <= 0xf3:
            return 3
    elif state in (1, 2, 3):
        if 0x80 <= byte <= 0xbf:
            return state - 1
    elif state == 4 and 0xa0 <= byte <= 0xbf:
        return 1  # no overlong three-byte form
    elif state == 5 and 0x80 <= byte <= 0x9f:
        return 1  # no surrogates
    elif state == 6 and 0x90 <= byte <= 0xbf:
        return 2  # no overlong four-byte form
    elif state == 7 and 0x80 <= byte <= 0x8f:
        return 2  # <= U+10FFFF
    return None


_UTF8_TRANSITIONS = tuple(tuple((byte + 1, next_state) for byte in range(256)
                                if (next_state := utf8_next_state(state, byte)) is not None)
                          for state in range(8))
_BYTE_VALUES = tuple(bytes([byte]) for byte in range(256))
_NEG_INF = float("-inf")


def _logadd(left, right):
    if left == _NEG_INF:
        return right
    if right == _NEG_INF:
        return left
    high, low = (left, right) if left >= right else (right, left)
    return high + math.log1p(math.exp(low - high))


def _validate_decoder_options(decode_mode, beam_width, max_decode_seconds):
    if decode_mode not in ("greedy", "utf8_fallback"):
        raise ValueError("decode_mode must be greedy or utf8_fallback")
    if type(beam_width) is not int or not 1 <= beam_width <= 256:
        raise ValueError("beam_width must be an integer in [1,256]")
    if type(max_decode_seconds) not in (int, float) or not math.isfinite(max_decode_seconds) or max_decode_seconds <= 0:
        raise ValueError("max_decode_seconds must be finite and positive")


def utf8_ctc_prefix_beam_search(log_probs, *, beam_width=8, max_decode_seconds=5.0, return_debug=False):
    """CPU CTC prefix beam on valid frames [T,257], no LM/length penalty/top-k.

    Every legal byte extension is considered. Blank and non-emitting repeats are
    handled separately, even if repeating the last byte is not a legal UTF-8
    extension. Paths merging into a prefix are SUMMED before beam pruning.
    Finite beam pruning is approximate, not an exact MAP guarantee.
    Transfer the CPU matrix once, never per scalar in the Python loop. Time limit
    aborts the entire decode, never returns a truncated transcript.
    """
    _validate_decoder_options("utf8_fallback", beam_width, max_decode_seconds)
    start = time.perf_counter()
    if not isinstance(log_probs, torch.Tensor) or log_probs.ndim != 2 or log_probs.size(1) != 257:
        raise ValueError("Expected log probabilities [T,257]")
    if log_probs.dtype != torch.float32:
        raise ValueError("CTC beam log probabilities must be float32")
    cpu = log_probs.detach().to(device="cpu", dtype=torch.float32)
    if torch.isnan(cpu).any() or torch.isposinf(cpu).any():
        raise ValueError("Invalid nonfinite log probabilities")
    if cpu.numel() and (cpu > 1e-5).any():
        raise ValueError("Expected log probabilities, not positive logits")
    frames = cpu.tolist()
    processed = 0

    def result(text=None, reason=None, score=None, candidates=None):
        output = {"text": text, "reason": reason, "frames_processed": processed,
                  "num_frames": len(frames), "beam_width": beam_width, "token_pruning": "none",
                  "decode_seconds": time.perf_counter() - start}
        if return_debug:
            output["raw_log_score"] = score  # NOT calibrated confidence.
            output["complete_candidates"] = candidates or []
        return output

    # prefix -> [log P(blank), log P(nonblank), UTF-8 state]
    beam = {b"": [0.0, _NEG_INF, 0]}
    for frame in frames:
        if time.perf_counter() - start >= max_decode_seconds:
            return result(reason="decode_time_budget_exceeded")
        next_beam = {}
        for prefix, (p_blank, p_nonblank, state) in beam.items():
            if time.perf_counter() - start >= max_decode_seconds:
                return result(reason="decode_time_budget_exceeded")
            total = _logadd(p_blank, p_nonblank)
            same = next_beam.setdefault(prefix, [_NEG_INF, _NEG_INF, state])
            same[0] = _logadd(same[0], total + frame[0])
            last_token = prefix[-1] + 1 if prefix else None
            if last_token is not None:
                same[1] = _logadd(same[1], p_nonblank + frame[last_token])
            for token, new_state in _UTF8_TRANSITIONS[state]:
                source = p_blank if token == last_token else total
                score = source + frame[token]
                if score == _NEG_INF:
                    continue
                extended = prefix + _BYTE_VALUES[token - 1]
                entry = next_beam.setdefault(extended, [_NEG_INF, _NEG_INF, new_state])
                entry[1] = _logadd(entry[1], score)
        # Merge before pruning; deterministic byte-prefix tie break.
        ranked = sorted(((prefix, values) for prefix, values in next_beam.items()
                         if _logadd(values[0], values[1]) != _NEG_INF),
                        key=lambda item: (-_logadd(item[1][0], item[1][1]), item[0]))
        beam = dict(ranked[:beam_width])
        processed += 1
        if not beam:
            return result(reason="no_valid_candidate")
    if time.perf_counter() - start >= max_decode_seconds:
        return result(reason="decode_time_budget_exceeded")
    complete = []
    for prefix, (p_blank, p_nonblank, state) in beam.items():
        if state == 0:
            text = prefix.decode("utf-8", errors="strict")
            complete.append((text, prefix, _logadd(p_blank, p_nonblank)))
    candidates = [{"text": text, "bytes_hex": prefix.hex(), "raw_log_score": score}
                  for text, prefix, score in complete] if return_debug else None
    for text, _, score in complete:
        if text.strip():
            return result(text=text, score=score, candidates=candidates)
    return result(reason="no_nonempty_candidate" if complete else "no_complete_utf8_candidate", candidates=candidates)


def decode_asr_logits(logits, *, decode_mode="greedy", beam_width=8, return_debug=False, max_decode_seconds=5.0):
    """Decode one already length-trimmed [T,257] utterance; no extra forward pass."""
    _validate_decoder_options(decode_mode, beam_width, max_decode_seconds)
    if logits.ndim != 2 or logits.size(1) != 257:
        raise ValueError("Expected valid-frame logits [T,257]")
    base = {"decoder_used": "greedy", "fallback_used": False}
    if not torch.isfinite(logits).all():
        return {**base, "status": "nonfinite_output", "text": None}
    path = logits.argmax(-1).tolist()
    raw = ctc_path_to_bytes(path)
    diagnostic = {"num_frames": len(path), "byte_length": len(raw), "token_path": path,
                  "raw_bytes_hex": raw.hex()} if return_debug else None
    try:
        text = normalize_text(raw.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as error:
        output = {**base, "status": "invalid_utf8", "text": None, "greedy_status": "invalid_utf8"}
        if return_debug:
            diagnostic["first_utf8_error"] = {"start": error.start, "end": error.end, "reason": error.reason,
                                                "offset_unit": "raw_byte", "end_exclusive": True}
            output.update(debug_text=raw.decode("utf-8", errors="replace"), decode_diagnostics=diagnostic)
        if decode_mode == "greedy":
            return output
        output.update(fallback_used=True, decoder_used="utf8_ctc_prefix_beam")
        beam = utf8_ctc_prefix_beam_search(logits.float().log_softmax(-1), beam_width=beam_width,
                                         max_decode_seconds=max_decode_seconds, return_debug=return_debug)
        if return_debug:
            diagnostic["beam"] = beam
        if beam["text"] is not None:
            text = normalize_text(beam["text"])
            if text.strip():
                output.update(status="ok", text=text)
                return output
        output["fallback_reason"] = beam["reason"] or "no_nonempty_candidate"
        return output
    status = "ok" if text.strip() else "empty_transcript"
    output = {**base, "status": status, "text": text, "greedy_status": status}
    if return_debug:
        output["decode_diagnostics"] = diagnostic
    return output


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
    def transcribe(self, waveform, sample_rate: int, request_id=None, *, decode_mode="greedy",
                   beam_width=8, return_debug=False, max_decode_seconds=5.0) -> dict:
        """Same acoustic forward. Fallback repairs decoding, not model accuracy.
        No VAD is implied; debug_text is never an official transcript.
        """
        _validate_decoder_options(decode_mode, beam_width, max_decode_seconds)
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
        decoded = decode_asr_logits(logits, decode_mode=decode_mode, beam_width=beam_width,
                                    return_debug=return_debug, max_decode_seconds=max_decode_seconds)
        if decoded.get("greedy_status") == "invalid_utf8":
            LOGGER.warning("Invalid greedy UTF-8 for request %s; final status=%s", request_id, decoded["status"])
        return {**base, **decoded}


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


def transcribe(model: ByteCTCASR, waveform, sample_rate: int, request_id=None, *, decode_mode="greedy",
               beam_width=8, return_debug=False, max_decode_seconds=5.0):
    return model.transcribe(waveform, sample_rate, request_id, decode_mode=decode_mode,
                            beam_width=beam_width, return_debug=return_debug, max_decode_seconds=max_decode_seconds)
