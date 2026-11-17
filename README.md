# MOT17 paper baselines

Chạy hai phương pháp tiền xử lý trên **cùng điều kiện MOT17/x265**:

- **SCOPE:** dùng mạng và hàm loss hỗ trợ từ mã chính thức, pin commit và kiểm hash; thích ứng sang MOT17/x265.
- **Zhao et al.:** tự cài đặt theo *A Preprocessing Framework for Video Machine Vision under Compression*; ghi rõ các giả định kiến trúc, codec và task loss.

**Trạng thái:** đã xác minh hiệu chuẩn entropy trên GPU Tesla T4 với 256 train clip. Hai canary đã được gửi lên Kaggle riêng tư. Chưa có benchmark AP/BD-Rate hoàn chỉnh. Mã này không phải bản tái lập các số COCO/GOT-10K gốc của hai bài.

## Điều kiện chung

| Hạng mục | Cấu hình |
|---|---|
| Train / validation / test | MOT17 02,04,05,10 / 09 / 11,13; chỉ bản FRCNN |
| Đầu vào | RGB 512×512, clip4; train stride2, test stride4 |
| Test | 1.648 frame vật lý; bỏ hai frame cuối của chuỗi13 để giữ clip đủ |
| Seeds | 2026,2027,2028; mỗi phương pháp train độc lập ba lần |
| Teacher | Frozen Faster R-CNN ResNet50-FPN v2; YOLOv8s chỉ khi eval |
| Tối ưu | Adam1e-4, batch2, tích lũy2, trần36 epoch, min12, patience10 |
| GPU / môi trường | Tesla T4, Python3.11.14, torch2.5.1, torchvision0.20.1 |
| Test codec | x265 medium, CRF22/27/32/37, IPPP clip4,25fps, aq-mode2 |
| Đánh giá | AP50/AP50:95 toàn tập, payload bpp, PCHIP BD-Rate và cubic sensitivity |

Hai phương pháp giữ loss riêng. Mục tiêu là đối chiếu dưới cùng điều kiện ngoài; không đổi cả hai sang một loss chung. Đây là **phát hiện người**, chưa phải benchmark tracking HOTA/MOTA/IDF1.

## Cài đặt

Python3.11 hoặc3.12. GPU thật và FFmpeg có `libx265`/`ffprobe` cần cho thí nghiệm; test đơn vị chạy được bằng CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Trên Windows, kích hoạt bằng `.venv\Scripts\Activate.ps1`. Chọn wheel PyTorch CPU/CUDA phù hợp máy trước khi cài requirements.

Tải đúng mã SCOPE chính thức trước khi test/train:

```bash
git clone https://github.com/miccunifi/SCOPE.git .runtime/SCOPE
git -C .runtime/SCOPE checkout --detach 453a40bebd73d4a7c3ad4436370e96f412a0a546
python -m pytest -q
```

Repo không kèm ảnh MOT17, PDF, credentials, trọng số detector hoặc checkpoint hiện có. MOT17 cần có `img1/`, `gt/gt.txt` và đầy đủ các chuỗi theo split bên trên.

## Hiệu chuẩn, train và eval cục bộ

Hiệu chuẩn một lần trên train, rồi dùng **cùng file và SHA256** cho toàn bộ job:

```bash
python scripts/run_paper_baselines.py --mode calibrate \
  --data-root /data/MOT17/train --output-dir outputs/calibration
```

Tạo ma trận sáu job, gắn hash entropy và source:

```python
import hashlib, json
from pathlib import Path
from scripts.run_paper_baselines import make_job
from scripts.kaggle_paper_baselines import pack_sources

root = Path.cwd()
digest = hashlib.sha256(Path('outputs/calibration/shared_entropy.pt').read_bytes()).hexdigest()
sources = pack_sources(root)['hashes']
Path('outputs/jobs').mkdir(parents=True, exist_ok=True)
for method in ('scope_official', 'zhao_reproduction'):
    for seed in (2026, 2027, 2028):
        job = make_job(method, seed, entropy_sha256=digest, source_sha256=sources)
        Path('outputs/jobs', job['job_id'] + '.json').write_text(json.dumps(job, indent=2))
```

```bash
python scripts/run_paper_baselines.py --mode train \
  --job-file outputs/jobs/scope_official_seed2026.json \
  --data-root /data/MOT17/train --output-dir outputs/scope-seed2026 \
  --entropy outputs/calibration/shared_entropy.pt
```

Đổi job-file/output-dir cho từng method/seed. Dùng `--mode canary` để kiểm128 optimizer update trước khi train. Hết phiên7,8h là trạng thái chưa hoàn thành; tiếp tục bằng `--resume <checkpoint.last.pt>` vào thư mục output mới, giữ nguyên job và artifact.

Chỉ eval checkpoint `completed`/`converged`, dùng best weights được chọn trên validation:

```bash
python -m pip install 'ultralytics==8.3.230' --constraint requirements.txt
python scripts/run_paper_baselines.py --mode eval \
  --job-file outputs/jobs/scope_official_seed2026.json \
  --checkpoint outputs/scope-seed2026/scope_official_seed2026.last.pt \
  --data-root /data/MOT17/train --output-dir outputs/eval/scope-seed2026
python scripts/merge_paper_baselines.py \
  --input-dir outputs/eval --output outputs/benchmark.json
```

Merger yêu cầu đủ sáu job và cùng dữ liệu/source/detector/codec/runtime. `--allow-partial` chỉ dành cho báo cáo tiến độ. AP không đơn điệu, plateau hoặc thiếu miền giao được ghi unavailable; không ngoại suy hoặc sửa đường cong để ép ra BD-Rate.

## Kaggle

```bash
python -m pip install -r requirements-kaggle.txt
python scripts/kaggle_paper_baselines.py --help
```

Launcher nhận manifest cục bộ và `--pool-file /private/pool.json`; pool có dạng `{username: access_token}` và không được commit. Worker chỉ đóng gói ba file runtime cho baseline; các dependency nền được checkout từ revision T-SCOPE đã pin. `export_approved` phải được người chạy đặt rõ trong manifest trước submission. Chỉ tạo kernel riêng tư mới; không ghi đè notebook cũ. Production bị chặn tới khi cả hai canary được tải về và xác minh.

```bash
python scripts/kaggle_paper_baselines.py status \
  --manifest /private/experiment-manifest.json --pool-file /private/pool.json
```

Repo không có credentials hoặc manifest cá nhân. Xem giao diện `build_worker(config)` để tạo worker từ job đã pin; kernel sources/datasets riêng tư phải có quyền truy cập hợp lệ. Chia sẻ artifact giữa tài khoản cần sự cho phép riêng của chủ tài khoản.

## Mã và tài liệu

| File | Vai trò |
|---|---|
| `models/paper_baselines.py` | Adapter SCOPE chính thức; mạng Zhao và loss riêng |
| `video_codecs/zhao_virtual_codec.py` | Prediction, DCT/STE và factorized CDF chung |
| `scripts/run_paper_baselines.py` | Hiệu chuẩn, canary, train/resume và eval |
| `scripts/kaggle_paper_baselines.py` | Worker pin nguồn, kiểm hash, quota/workload, submission riêng tư |
| `scripts/merge_paper_baselines.py` | Kiểm dữ liệu ghép cặp và tổng hợp AP/bpp/BD-Rate |
| `tests/test_paper_baselines.py` | Gradient, temporal behavior, provenance, admission và merge |

[Protocol và giả định](docs/PAPER_BASELINES_MOT17.md) · [Thiết kế đã duyệt](docs/superpowers/specs/2026-11-17-matched-paper-baselines-design.md) · [Bằng chứng hiệu chuẩn](docs/calibration_verification.json) · [Nguồn và giấy phép](THIRD_PARTY_NOTICES.md).

Các module T-SCOPE còn lại là dependency tương thích cho dataset, codec thật, teacher, metrics và checkpoint; không phải baseline thứ ba đã được train trong repo này.
