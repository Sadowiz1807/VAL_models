# Đặc tả kiến trúc Model 1 — Audio → Waveform → ASR → Text

Tài liệu liên quan: [Model 2 — Text → Act, Goal, Action, Parameters và Confidence]("NLUModel_architecture.md").

## 1. Phạm vi và luồng xử lý

Luồng inference:

`Audio → Waveform → Log-Mel → Subsampling → Conformer → CTC → Text`

Model 1 chịu trách nhiệm chép lại lời nói. Model 2 tiếp nhận text, xác định act, goal/domain, action, parameters và quan hệ tác vụ. Khi tách model, bỏ các head semantic khỏi đường chạy ASR; không dùng semantic loss để nói rằng transcript chắc chắn đúng.

Cấu hình cơ sở sử dụng log-Mel 80 chiều, Conformer 8 lớp × 512 chiều và vocabulary UTF-8 byte gồm 257 phần tử. Các giá trị được kế thừa từ cấu hình acoustic tham chiếu; chất lượng huấn luyện phải được xác lập bằng thực nghiệm. Checkpoint ASR bên ngoài phải được sử dụng cùng frontend, tokenizer và kiến trúc tương thích, hoặc được chuyển đổi bằng quy trình migration có kiểm chứng.

## 2. Cấu hình thành phần

| Khóa config | Giá trị cơ sở | Vai trò trong Model 1 |
| --- | --- | --- |
| `audio.sample_rate`, `channels` | 16000, 1 | Waveform mono 16 kHz |
| `audio.frontend`, `feature_dim` | `log_mel`, 80 | Input acoustic `[B,T,80]` |
| `n_fft`, `win_length`, `hop_length` | 400, 400, 160 | Cửa sổ 25 ms, bước 10 ms |
| `f_min`, `f_max` | 0, 8000 Hz | Miền tần số Mel |
| `speech_encoder.d_model`, `layers` | 512, 8 | Hidden size và số block |
| `attention_heads`, `d_ff` | 8, 2048 | Mỗi attention head có 64 chiều |
| `dropout`, `conv_kernel_size` | 0.1, 31 | Regularization và convolution trong Conformer |
| `subsampling_factor` | 4 | Giảm số frame; cấu hình tham chiếu không định nghĩa implementation |
| `lexical_branch.type`, `vocab_size` | `ctc_utf8_bytes`, 257 | CTC output head |
| `lexical_branch.blank_id` | 0 | Blank CTC |
| `semantic_resampler` | 32 latent × 512 | Không cần trong ASR tách riêng |
| `semantic_core`, `operation_decoder` | 4 lớp; tối đa 4 operations | Thành phần semantic thuộc Model 2 |

`get_config()` tạo dữ liệu cấu hình. Tính đúng đắn của subsampling, Conformer, padding, initialization và gradient checkpointing trong `nn.Module` phải được kiểm chứng độc lập ở implementation.

## 3. Hợp đồng dữ liệu và tensor

Ký hiệu: `B` là batch size, `N` là số mẫu waveform, `T` là số frame Mel, `T'` là số frame sau subsampling, `U` là số byte target, `D=512`.

| Điểm trong pipeline | Tensor/kiểu | Ghi chú |
| --- | --- | --- |
| Waveform một mẫu | float32 `[N]` | Biên độ PCM được scale đúng; mono |
| Log-Mel một mẫu | float32 `[T,80]` | Trích đặc trưng từng audio trước khi pad |
| Batch log-Mel | `[B,T_max,80]` | Có `mel_lengths: int64[B]` |
| Subsampling | `[B,T'_max,512]` | Có `encoder_lengths: int64[B]` |
| Speech encoder | `[B,T'_max,512]` | Mask padding |
| CTC logits | `[B,T'_max,257]` | Logits chưa softmax |
| Input `CTCLoss` | float32 `[T'_max,B,257]` | `log_softmax` rồi transpose |
| CTC target | int64 `[sum(U_i)]` | Ghép targets, không có blank |
| Output cuối | UTF-8 string | Không chứa act/goal |

Với frontend cơ sở có `center=False`, `pad=0`, `N>=400`:

```text
T = floor((N - 400) / 160) + 1
T' = ceil(ceil(T / 2) / 2) = ceil(T / 4)
```

Ví dụ 5 giây, 80.000 mẫu: `T=498`, `T'=125`. Công thức chỉ đúng cho frontend và hai convolution stride-2 được mô tả bên dưới. Không dùng nó nếu implementation thực tế dùng `center=True`, convolution không padding hoặc một frontend khác.

## 4. Vocabulary UTF-8 byte và quy tắc giải mã

`default_lexical_vocab()` trong cấu hình tham chiếu tạo ánh xạ:

```text
ID 0        = <BLANK>
ID 1..256   = <BYTE:00> .. <BYTE:ff>
token_id    = byte_value + 1
```

Đây là vocabulary cố định; không cần học bộ token tiếng Việt bằng BPE để sử dụng nhánh này. Một ký tự tiếng Việt có thể cần nhiều byte UTF-8. Biểu diễn được mọi text UTF-8 không đồng nghĩa ASR nhận dạng chính xác mọi tiếng Việt.

Mã tham chiếu cho mã hóa target và giải mã CTC:

```python
import unicodedata

def encode_utf8_target(text: str) -> list[int]:
    text = unicodedata.normalize("NFC", text)
    return [byte + 1 for byte in text.encode("utf-8")]

def min_ctc_frames(ids: list[int]) -> int:
    adjacent_repeats = sum(a == b for a, b in zip(ids, ids[1:]))
    return len(ids) + adjacent_repeats

def decode_ctc_path(path: list[int]) -> str:
    kept = []
    previous = None
    for token in path:
        if not 0 <= token <= 256:
            raise ValueError("CTC token outside byte vocabulary")
        if token != previous and token != 0:
            kept.append(token)
        previous = token  # blank cũng phải cập nhật previous
    return bytes(token - 1 for token in kept).decode("utf-8", errors="strict")
```

Thứ tự decode bắt buộc: **gộp các token lặp liên tiếp trên đường dự đoán, rồi loại blank**. Ví dụ đường `[98,0,98]` phải thành `aa`, không phải `a`.

Byte sequence dự đoán có thể không hợp lệ UTF-8. Baseline bắt `UnicodeDecodeError`, ghi log và trả lỗi decode/hỏi nói lại; không âm thầm xóa byte lỗi rồi xem đó là transcript tin cậy. Sau này có thể dùng beam search có ràng buộc UTF-8.

### 4.1. Điều kiện căn chỉnh CTC

Với từng mẫu, cần `T' >= min_ctc_frames(target_ids)`. Byte targets dài hơn character targets, trong khi subsampling 4x chỉ giữ khoảng 25 frame/giây. Tỷ lệ mẫu không căn chỉnh được phải được thống kê trên dataset trước khi đánh giá tính phù hợp của cấu hình.

Nếu nhiều mẫu không đủ frame:

1. Kiểm tra alignment, transcript thừa và audio bị cắt.
2. Thử subsampling 2x để tăng số frame, đo lại VRAM/latency.
3. So sánh character/subword CTC trên cùng dữ liệu, đồng thời thay output head và target codec tương ứng.

Không dùng `zero_infinity=True` như một cách che toàn bộ mẫu lỗi. CTC phải nhận target không có blank và độ dài đúng; tài liệu PyTorch mô tả các điều kiện input/target. [R1]

## 5. Tổ chức mã nguồn

Mã nguồn của Model 1 trong thư mục `asr/` gồm đúng ba file:

| File | Class/hàm cần chứa |
| --- | --- |
| `asr/config.py` | `create_config()`, `load_config()`, `save_config()`, `validate_config()`; audio/encoder/codec/training config và version |
| `asr/model.py` | Audio loading/resample dùng chung, `LogMelFrontend`, byte codec, `ByteCTCASR`, `build_model()`, `load_for_inference()`, `transcribe()` |
| `asr/train.py` | `ASRDataset`, `collate_asr`, `compute_ctc_loss`, `train_one_epoch`, `validate`, `fit`, checkpoint/optimizer/scheduler, metrics và CLI |

Không tạo thêm file Python riêng cho frontend, codec, dataset, loss, inference hoặc evaluate. Dùng class/hàm và các phần rõ ràng bên trong ba file. Config JSON, tokenizer, weights và log là tài nguyên; chúng không làm tăng số file `.py`.

Phụ thuộc: `model.py` dùng `config.py`; `train.py` dùng cả `config.py` và `model.py`. `model.py` không import `train.py`; `config.py` không import hai file còn lại. Frontend/codec chỉ có một implementation trong `model.py`, được train và inference dùng chung.

Để giữ đúng ba file, hai thư mục `asr/` và `nlu/` có thể dùng namespace package của Python 3, không cần `__init__.py`. Chạy từ thư mục cha bằng `python -m asr.train` hoặc `python -m nlu.train`; imports nội bộ dùng `from .config import ...`, `from .model import ...`. Orchestration giữa các model nằm ngoài phạm vi ba file mã nguồn của từng model.

Phân bổ mã tham chiếu: codec ở mục 4 và các lớp mục 6 thuộc `model.py`; collate/loss và training ở mục 7–9 thuộc `train.py`; mọi mặc định và validator cấu hình thuộc `config.py`. Hàm validate dữ liệu/model quality ở `train.py` khác hàm validate cấu hình ở `config.py`.

### 5.1. API phía sử dụng

`load_for_inference(checkpoint_path, device)` dựng model từ config nằm trong checkpoint, nạp weights tương thích và chuyển `eval`. `transcribe(waveform, sample_rate)` dùng đúng frontend/codec đã version hóa. Inference sử dụng `asr.model` và độc lập với Dataset, optimizer và vòng huấn luyện.

## 6. Frontend và acoustic model

Mã tham chiếu sử dụng API PyTorch/Torchaudio 2.8 [R2–R3]. Môi trường huấn luyện phải cố định cặp phiên bản tương thích CUDA/CPU. Quá trình khởi tạo trong mã tham chiếu tạo trọng số mới và không tự tải pretrained weights.

### 6.1. Frontend

```python
import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram

class LogMelFrontend(nn.Module):
    def __init__(self, audio_cfg):
        super().__init__()
        self.sample_rate = audio_cfg["sample_rate"]
        self.n_fft = audio_cfg["n_fft"]
        self.mel = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=audio_cfg["win_length"],
            hop_length=audio_cfg["hop_length"],
            f_min=audio_cfg["f_min"], f_max=audio_cfg["f_max"],
            n_mels=audio_cfg["feature_dim"],
            center=False, pad=0, power=2.0,
            norm=None, mel_scale="htk",
        )

    def forward(self, mono_waveform, sample_rate):
        if mono_waveform.ndim != 1:
            raise ValueError("Expected mono [N]")
        if sample_rate != self.sample_rate:
            raise ValueError("Resample waveform before frontend")
        if mono_waveform.numel() < self.n_fft:
            raise ValueError("Audio too short")
        if not torch.isfinite(mono_waveform).all():
            raise ValueError("Non-finite waveform")
        mel = self.mel(mono_waveform.float())
        return mel.clamp_min(1e-5).log().transpose(0, 1)  # [T,80]
```

Cấu hình tham chiếu V1 không xác định `center`, `power`, `mel_scale`, `norm`, log base hoặc CMVN. Các giá trị được sử dụng trong frontend phải được lưu trong config/checkpoint. Khi sử dụng pretrained frontend, các giá trị này phải tương thích với quá trình tiền huấn luyện.

Collator gọi frontend từng waveform, pad feature bằng 0 ở cuối và lưu độ dài thật. Nếu thêm CMVN, tính thống kê trên frame hợp lệ; không cho padding tham gia. Frontend và input cần ở cùng device.

### 6.2. Acoustic model

```python
import torch
from torch import nn
from torchaudio.models import Conformer

def time_padding_mask(lengths, width):
    return torch.arange(width, device=lengths.device)[None, :] >= lengths[:, None]

class ByteCTCASR(nn.Module):
    def __init__(self, config):
        super().__init__()
        s = config["speech_encoder"]
        d = s["d_model"]
        if s["subsampling_factor"] != 4:
            raise ValueError("This skeleton implements exactly two stride-2 layers")
        self.subsample = nn.ModuleList([
            nn.Conv1d(config["audio"]["feature_dim"], d, 3, stride=2, padding=1),
            nn.Conv1d(d, d, 3, stride=2, padding=1),
        ])
        self.encoder = Conformer(
            input_dim=d, num_heads=s["attention_heads"],
            ffn_dim=s["d_ff"], num_layers=s["layers"],
            depthwise_conv_kernel_size=s["conv_kernel_size"],
            dropout=s["dropout"], use_group_norm=True,
        )
        self.ctc_head = nn.Linear(d, config["lexical_branch"]["vocab_size"])

    def forward(self, mels, lengths):
        # mels [B,T,80]; lengths [B], cùng device
        if (lengths <= 0).any() or (lengths > mels.size(1)).any():
            raise ValueError("Invalid Mel lengths")
        pad = time_padding_mask(lengths, mels.size(1))
        x = mels.masked_fill(pad[..., None], 0).transpose(1, 2)
        for conv in self.subsample:
            x = torch.nn.functional.gelu(conv(x))
            lengths = (lengths + 1) // 2
            pad = time_padding_mask(lengths, x.size(-1))
            x = x.masked_fill(pad[:, None, :], 0)
        h, lengths = self.encoder(x.transpose(1, 2), lengths)
        return {"logits": self.ctc_head(h), "lengths": lengths}
```

Implementation subsampling tham chiếu gồm hai Conv1d. Giá trị `subsampling_factor=4` không đủ để xác định tính tương thích của implementation; cấu trúc acoustic phải được đối chiếu trước khi nạp weights cũ. Mã tham chiếu sử dụng `use_group_norm=True`. Normalization theo thời gian có thể chịu ảnh hưởng của padding; dữ liệu nên được bucket theo độ dài và kiểm tra tính ổn định của dự đoán khi thay mức padding trong batch.

`gradient_checkpointing=True` trong config không tự làm `nn.Module` checkpoint activation. Nếu cần tính năng này, phải thêm implementation tương ứng và kiểm tra gradient/recompute.

## 7. Collate, loss và cập nhật trọng số

```python
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

def collate_asr(feature_and_target_pairs):
    features, targets = zip(*feature_and_target_pairs)
    if any(len(t) == 0 for t in targets):
        raise ValueError("Baseline ASR training requires nonempty transcripts")
    return {
        "mels": pad_sequence(features, batch_first=True, padding_value=0.0),
        "mel_lengths": torch.tensor([len(x) for x in features], dtype=torch.long),
        "targets": torch.tensor([v for y in targets for v in y], dtype=torch.long),
        "target_lengths": torch.tensor([len(y) for y in targets], dtype=torch.long),
        "min_frames": torch.tensor([min_ctc_frames(y) for y in targets], dtype=torch.long),
    }

ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)

def compute_ctc_loss(output, batch):
    logits, lengths = output["logits"], output["lengths"]
    if (lengths < batch["min_frames"].to(lengths.device)).any():
        raise ValueError("CTC alignment impossible: inspect sample IDs")
    log_probs = logits.float().log_softmax(-1).transpose(0, 1)
    return ctc(log_probs, batch["targets"], lengths, batch["target_lengths"])
```

Trước khi gọi model/loss, chuyển batch tensor về device theo backend đã chọn. Xác minh yêu cầu device/dtype lengths của phiên bản CTC/CUDA đang dùng. Không đưa `IGNORE_INDEX=-100` vào CTC target; sentinel đó phù hợp các loss phân loại semantic hơn.

Với mixed precision, tính `log_softmax` và CTC ở float32. Khi dùng FP16 GradScaler: scale loss, backward, unscale optimizer trước `clip_grad_norm_`, sau đó step và update scaler.

Cấu hình tham chiếu sử dụng `batch_size=16`, `gradient_accumulation_steps=3`: effective batch khoảng 48 mẫu trên một thiết bị nếu đủ ba microbatch. Nhóm cuối epoch có ít microbatch phải được scale theo số thực hoặc xử lý rõ. `learning_rate=1e-4`, `weight_decay=0.01`, `gradient_clip_norm=1.0`, `max_epochs=100` là cấu hình đầu vào, không phải tham số đã tối ưu.

## 8. Giai đoạn huấn luyện và migration

| Stage cũ | `enabled_losses` | Xử lý với Model 1 mới |
| --- | --- | --- |
| `ACOUSTIC` | `lexical` | Đổi thành stage ASR chính |
| `BRIDGE` | `bridge_alignment` | Không dùng khi interface production là text |
| `SEMANTIC` | Các loss semantic | Sang Model 2 |
| `PARAMETER` | `parameters` | Sang Model 2 |
| `SAFETY` | `ood` | Sang Model 2 nếu có detector đó |
| `JOINT` | Semantic + lexical | Tách; không còn một graph chung qua text |

Khi ASR là model riêng, dùng `L_ASR = L_CTC` làm loss cơ sở. Hệ số `lexical=0.25` trong cấu hình tham chiếu V1 là trọng số đa nhiệm; không bắt buộc giữ 0.25 khi loss chỉ còn CTC. Việc đổi scale loss cần theo dõi optimizer, gradient norm và clipping, không mặc định hoàn toàn tương đương.

Quy trình build:

1. Trong `asr/config.py`, viết `create_config()` có version riêng, chứa audio, speech encoder, lexical branch và ASR training.
2. Viết validator riêng; không đưa config đã bỏ ontology/semantic qua nguyên `validate_config()` V1.
3. Tạo manifest audio–transcript, thống kê sample rate, độ dài và điều kiện căn chỉnh byte CTC.
4. Viết codec/frontend trong `asr/model.py`, collator/loss/validation trong `asr/train.py`; kiểm tra trước khi train.
5. Build `ByteCTCASR`; nếu có checkpoint cũ, so tên key và shape, chỉ chuyển weights tương thích có báo cáo.
6. Chạy một batch forward/backward, overfit một tập nhỏ để kiểm tra khả năng học.
7. Train đầy đủ, chọn checkpoint bằng validation CER/WER và lỗi thực thể.
8. Đóng version ASR, tạo transcript lỗi thực cho quá trình train Model 2.

Không dùng `load_state_dict(strict=False)` rồi bỏ qua toàn bộ missing/unexpected keys. Architecture name và tensor shape giống nhau chưa chứng minh preprocessing hoặc token-ID semantics giống nhau.

## 9. Dữ liệu và tiêu chí đánh giá

Một mẫu train tối thiểu:

```json
{
  "dataset_id": "local_vi_commands",
  "sample_id": "utt-001",
  "audio_ref": "audio/utt-001.wav",
  "transcript": "mở Chrome",
  "speaker_id": "spk-01",
  "session_id": "session-01"
}
```

Trong cấu hình tham chiếu, `data.required_fields` gồm `dataset_id`, `sample_id`, `audio_ref`; `transcript_required_for` quy định ENTITY/FREE_TEXT. Đối với ASR standalone supervised, transcript là trường bắt buộc của mọi mẫu sử dụng CTC, kể cả mẫu semantic không có entity.

Chia speaker/session và nhóm audio gốc trước augmentation. Calibration/test của Model 2 không được lọt vào corpus train ASR của cặp checkpoint đang được đánh giá. Tách test câu mới, speaker mới, tên riêng và tiếng ồn; không chỉ chấm câu template đã gặp.

| Metric | Cách dùng |
| --- | --- |
| CER | Giữ Unicode normalization cố định; chấm lỗi dấu |
| WER | Công bố tách khoảng trắng hay word segmentation tiếng Việt |
| Exact match thực thể | Chấm tên ứng dụng, URL, command alias, query, số |
| Lỗi từ quyết định | “không/đừng”, “mở/đóng”, “tăng/giảm” |
| Invalid UTF-8 rate | Tỷ lệ đường decode byte lỗi |
| Infeasible CTC rate | Tỷ lệ target không đủ frame trước train |
| RTF, latency p50/p95 | Cố định device/batch; tách frontend, encoder, decode |
| End-to-end exact match | Act + tất cả operations/actions/parameters đúng theo audio gốc |

Confidence của Model 2 cao trên text sai không sửa được thông tin ASR đã mất. Vì vậy metric end-to-end bắt buộc dùng yêu cầu đúng từ audio gốc, không dùng transcript ASR làm ground truth.

## 10. Inference và biên giữa hai model

```text
read/record audio
→ check channels / sample rate / finite values
→ resample thật về 16 kHz
→ extract feature bằng cùng frontend version
→ model.eval() + inference_mode()
→ lấy path trong encoder_lengths của từng mẫu
→ CTC collapse + UTF-8 decode
→ text normalization đã version hóa
→ gọi Model 2 bằng text
```

Envelope đầu ra tham chiếu:

```json
{
  "request_id": "demo-001",
  "asr_version": "asr-draft-1",
  "status": "ok",
  "text": "mở Chrome"
}
```

Trạng thái lỗi input, audio quá ngắn, không có speech hoặc lỗi UTF-8 được xử lý riêng; không biến lỗi kỹ thuật thành text “không hỗ trợ”. Kiểm tra no-speech cần cơ chế riêng được đánh giá; một CTC decoder không tự bảo đảm việc này.

## 11. Tối ưu hai model và hướng mở rộng

Việc thêm act, goal/domain, action hoặc parameter slot thuộc Model 2. Model 1 vẫn trả text, nên không phải đổi CTC output head chỉ vì số act/domain tăng. Chỉ fine-tune ASR thêm nếu miền mới chứa âm thanh/từ vựng/tên riêng mà model nhận dạng chưa tốt; byte vocabulary hiện tại vẫn có thể biểu diễn text UTF-8 đó.

**Giữ interface text:** train ASR, đóng checkpoint, sinh transcript dự đoán trên tập thích hợp, train Model 2 với text sạch + text lỗi ASR, rồi đo cặp checkpoint. Đây là tối ưu phối hợp, không phải gradient end-to-end.

**Joint training thực sự:** phải thêm đường khả vi từ speech representations/distributions sang semantic model hoặc thiết kế policy-gradient cho lựa chọn rời rạc. Có thể giữ text cho quan sát nhưng inference không còn chỉ có text nếu semantic phụ thuộc hidden-state bridge. Đây là một nhánh kiến trúc khác, cần train/test như một hệ thống khác.

Hướng mở rộng theo thứ tự:

1. Đo byte CTC 4x, ưu tiên sửa dữ liệu và lỗi căn chỉnh.
2. So sánh 2x hoặc tokenizer khác nếu có bằng chứng byte target quá dài.
3. Beam search và ràng buộc UTF-8/từ vựng; không ép transcript vào action có sẵn.
4. Streaming với chunk/cache/causal attention; Conformer full-context hiện mô tả chưa là streaming chỉ nhờ chia file audio.
5. Distillation/quantization sau baseline; đo lại lỗi entity, độ trễ, bộ nhớ và pipeline.

## 12. Tiêu chí kiểm chứng

- Codec: tiếng Việt có dấu, byte lặp, blank xen kẽ, UTF-8 lỗi.
- Frontend: sample rate, N=400 và N<400, shape, finite output, padding.
- CTC: đúng target lengths, blank không nằm trong targets, điều kiện căn chỉnh.
- Model: shape logits, gradient không NaN, thử overfit tập nhỏ.
- Inference: cùng audio cho kết quả ổn định trong `eval`; so batch padding khác nhau.
- Hệ thống: lỗi mở/đóng, phủ định, số, tên ứng dụng; không phát lệnh khi ASR lỗi.

**Phạm vi kiểm chứng của mã tham chiếu:** cấu hình, phép đếm, cú pháp Python và các helper thuần Python đã được kiểm tra. Forward/backward của các `nn.Module` chưa được xác nhận bằng thực thi PyTorch/Torchaudio; kiểm chứng này là điều kiện trước khi triển khai.

## Tài liệu tham khảo

- **Cấu hình cơ sở:** `config(2).py` — nguồn tham chiếu cho các giá trị và giới hạn V1.
- [R1 — PyTorch 2.8: CTCLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CTCLoss.html).
- [R2 — Torchaudio 2.8: Conformer](https://docs.pytorch.org/audio/2.8/generated/torchaudio.models.Conformer.html).
- [R3 — Torchaudio 2.8: MelSpectrogram](https://docs.pytorch.org/audio/2.8/generated/torchaudio.transforms.MelSpectrogram.html).
