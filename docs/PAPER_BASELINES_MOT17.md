# Hai đối chứng bài báo trên cùng điều kiện MOT17

Trạng thái ngày 17/11/2026: người dùng đã cho phép rõ việc gửi mã lên
Kaggle riêng tư. **Hiệu chuẩn entropy đã hoàn thành** và artifact đã được
tải về kiểm hash/source/runtime. Hai canary đang được triển khai. Chưa có
kết quả AP/BD-Rate hoàn chỉnh; trạng thái cụ thể nằm trong manifest.

Artifact chung: `62edede3bf96bbe51bf8a5aef3bda08a17173dd069d63ebf058b05bef518b1e2`.
256 clip hiệu chuẩn, chỉ train02/04/05/10; Python3.11.14,
torch2.5.1+cu124, torchvision0.20.1+cu124, CUDA12.4, Tesla T4.

## Cấu hình đã được người dùng duyệt

| Điều kiện | Áp dụng cho cả hai |
|---|---|
| Train / validation / test | MOT17 02,04,05,10 / 09 / 11,13; chỉ bản FRCNN |
| Khung test | 1.648 khung vật lý; clip đủ 4 frame |
| Đầu vào | RGB 512×512, clip4; train stride2, test stride4 |
| Seed độc lập | 2026, 2027, 2028 |
| Teacher | Frozen Faster R-CNN R50-FPN v2; YOLOv8s chỉ khi đánh giá |
| Tối ưu | Adam 1e-4, weight decay0, batch2, tích lũy2, clip gradient1 |
| Ngân sách | Tối đa36 epoch; tối thiểu12, patience10; hết phiên7,8h thì resume |
| Thiết bị / môi trường | Tesla T4, Python3.11.14, torch2.5.1, torchvision0.20.1 |
| Codec khi train | Codec ảo dùng chung; cùng entropy prior hiệu chuẩn rồi đóng băng |
| Codec khi test | x265 medium, CRF22/27/32/37, IPPP,25fps, aq-mode2, không delta-QP |
| Chỉ số | AP50/AP50:95 toàn tập, payload bpp, PCHIP BD-Rate; cubic kiểm tra độ nhạy |

Đây là đánh giá **phát hiện người trên MOT17**, phù hợp protocol hiện có của
dự án; chưa triển khai chỉ số tracking HOTA/MOTA/IDF1.

Hai baseline khớp các điều kiện trong bảng với nhau. So với các job
T-SCOPE v3 đã train, dữ liệu và protocol đánh giá được giữ; optimizer
baseline là Adam1e-4 theo cấu hình vừa duyệt, trong khi V3 đã dùng
AdamW1.5e-4. So sánh với V3 phải công khai khác biệt huấn luyện này.

## Nguồn và các giả định cần giữ khi báo cáo

**SCOPE:** dùng nguyên mạng và hàm loss hỗ trợ từ
[miccunifi/SCOPE](https://github.com/miccunifi/SCOPE), commit
`453a40bebd73d4a7c3ad4436370e96f412a0a546`; xác minh hash trước import.
Giữ cấu hình loss của full launcher. Train lại từ đầu với một codec-family
x265, thay COCO bằng MOT17, teacher Detectron2 bằng teacher torchvision chung,
và thay proxy ảnh bằng proxy video chung. Không dùng trọng số COCO hay nhánh
`scope_paper_adapted` hiện có để giả làm kết quả mã chính thức.

**Zhao:** tự cài đặt theo PDF trong `D:/STUDY/LAB/News` và
[bài gốc](https://arxiv.org/abs/2512.15331). Sơ đồ Figure2 dùng16 kênh và xếp
RGB của4 frame theo chiều kênh; spatial residual block, conditional attention
và residual RGB. ReLU, một resblock hai convolution và sigmoid attention là
các giả định. Bốn frame có thể dùng thông tin tương lai trong cùng clip.
Loss giữ `10*(MSE toàn khung +0.001*estimated_bpp)+task_loss`; task_loss được
thích ứng thành normalized smooth-L1 FPN p2-p5 trên toàn khung. Không dùng nhánh
`proxy_only_zhao` cũ làm bản tái lập bài báo.

**Codec ảo:** DCT8×8, STE quantization, intra search trên DC palette0/0.5/1,
inter motion từ frame tái tạo trước. Ước lượng tính2 mode-bit/block8×8 và14
motion-bit/block16×16 của P-frame. Palette intra hẹp là một giới hạn của bản
cài đặt giả định. Separate I/P monotonic factorized CDF cho từng RGB/tần số,
fit một lần trên256 train clip, seed9090 rồi freeze. Không lấy frame test để
hiệu chuẩn. SCOPE giữ TV rate loss bản gốc; Zhao dùng CDF bpp theo loss bài báo.
Bitrate cuối cùng phải đo bằng x265 thật, không suy ra từ proxy.

Tên kết quả phải ghi **official-code SCOPE adapted to MOT17/x265** và
**independent Zhao reproduction adapted to MOT17/x265**. Các kết quả này không
phải số tái lập COCO/GOT-10K công bố trong hai bài.

## Kiểm chứng và thứ tự chạy

Kiểm tra workspace gốc: **181 passed,2 skipped,314 subtests passed**.
Bản xuất repo riêng: **19 passed**, bao gồm kiểm tra đường dẫn pool
có thể cấu hình thay cho đường dẫn riêng của máy nguồn.
Chưa kiểm chứng bộ nhớ/tốc độ T4 hoặc kết quả MOT17 thật.

1. Kiểm tra quota và workload sẵn có; tạo kernel mới, riêng tư, không ghi đè.
2. Hiệu chuẩn một entropy artifact và ghi hash dữ liệu, source, runtime.
3. Chạy canary128 optimizer update cho mỗi phương pháp với MOT17 thật; tải về
   kiểm tra checkpoint hữu hạn, protocol hash và đúng Tesla T4.
4. Chỉ sau khi cả hai canary đạt mới mở đủ sáu job train độc lập. Hết phiên
   giữ nguyên optimizer/scaler, thứ tự dữ liệu, lịch severity và tổng loss epoch.
5. Chỉ đánh giá checkpoint hoàn thành/hội tụ, được chọn trên validation.
6. Kiểm tra hash teacher giữa train/eval, hash YOLO và build ffmpeg/libx265;
   lưu raw detections và packet sizes. Merge chỉ final-ready khi đủ sáu job
   và cùng dữ liệu, cấu hình, source, detector, codec, runtime.

Mã: `models/paper_baselines.py`, `video_codecs/zhao_virtual_codec.py`,
`scripts/run_paper_baselines.py`, `scripts/kaggle_paper_baselines.py`,
`scripts/merge_paper_baselines.py`. Manifest/gói xem trước nằm trong
`outputs_v3/paper_baselines_20261117/`; không chứa token pool.

## Nội dung cần chuyển sang Kaggle

Worker mới mang đúng ba file mã: `models/paper_baselines.py`,
`video_codecs/zhao_virtual_codec.py`, `scripts/run_paper_baselines.py`, kèm cấu
hình và hash. Worker tự tải hai revision GitHub công khai và dataset MOT17 đã
có trên Kaggle. Không đưa pool credentials, PDF trong News, checkpoint hiện có
hoặc dữ liệu cục bộ vào worker. Các kernel được đặt riêng tư.
