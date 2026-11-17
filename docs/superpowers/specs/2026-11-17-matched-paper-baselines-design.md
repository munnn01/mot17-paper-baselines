# Matched MOT17 paper baselines

User request: run the official miccunifi/SCOPE implementation and independently
implement Zhao et al., *A Preprocessing Framework for Video Machine Vision under
Compression*, then run both under the same conditions. The user authorized the
Zhao implementation with explicit assumptions on 2026-11-17.

## Approach

Retrain both preprocessors from scratch on the project's existing MOT17 split.
Use the official SCOPE network at commit
`453a40bebd73d4a7c3ad4436370e96f412a0a546`, with its native spatial attention,
background replacement and loss terms. Implement Zhao from the local PDF,
including Figures 2 and 3. Existing `scope_paper_adapted` and
`proxy_only_zhao` are different internal controls and are not reused as these
paper baselines. Original COCO/GOT-10K checkpoints would introduce unmatched
training data; changing only the architecture while using the T-SCOPE objective
would not preserve the two papers' methods.

## Common conditions

- MOT17 train 02/04/05/10; validation 09; held-out evaluation 11/13.
- FRCNN copy only; no duplicate SDP/DPM videos; 1,648 held-out physical frames,
  excluding the last two frames of sequence 13 to retain complete clips.
- RGB 512x512, clip length 4, existing deterministic crop/clip order and paired
  severity schedule. Seeds 2026/2027/2028, three independent runs per method.
- Frozen pretrained Faster R-CNN ResNet50-FPN v2 for both methods. YOLOv8s is
  evaluation only. Pedestrian detection AP50/AP50:95, not MOT tracking scores.
- Both train through the same differentiable video codec: block intra/inter
  prediction, 8x8 DCT, straight-through quantization, factorized entropy prior.
  Calibrate the shared entropy model on training frames only, then freeze it.
- Both use Adam at 1e-4, batch 2 clips, accumulation 2, identical schedule,
  36-epoch maximum, minimum 12 epochs, patience 10 and min_delta 0.0005.
  Select best validation loss within each method; loss values across methods
  are not compared. A session deadline of 7.8 hours is a resumable interruption,
  not convergence. Both methods receive the same epoch/update ceiling.
- Same Python 3.11.14, torch 2.5.1/torchvision 0.20.1, GPU type (Tesla T4),
  precision policy and software versions. Reject an unexpected GPU type.
- Test uses actual x265, preset medium, CRF 22/27/32/37, 25 fps, IPPP clip4,
  aq-mode 2, no semantic delta-QP. CRF is explicitly not called fixed QP.
- Sum actual video packet payload bytes for bpp; same detector weights,
  thresholds, ignore-box policy and dataset-level AP implementation. PCHIP
  BD-Rate primary, cubic sensitivity; no extrapolation or forced monotonicity.

## Method-specific behavior and assumptions

SCOPE: import unchanged official `scope/model.py` and loss helpers, keep
48/96/192 channels and the full launcher settings (blur replacement kernel 51,
attention bias -4 and bounded object/background residuals). Train one x265
family instead of the paper's image codec mixture. Replace the original
Detectron2 teacher with the common torchvision teacher, and the original
image-codec proxies with the shared video proxy. These are declared adaptations.

Zhao: Figure 2 explicitly shows 16 channels and temporal RGB frames stacked
in channels before a 3x3 convolution. Use that design, not an invented 3D U-Net.
Use shared per-frame spatial convolution and a residual block, clip-level
temporal convolution, conditional attention fusion and residual RGB output.
The paper does not fix activation, residual-block count, attention internals,
search radius, DCT block size, entropy calibration or quantizer mapping. Declare
ReLU, one two-convolution residual block, sigmoid channel attention, block
search and orthonormal 8x8 DCT. The assumed intra search selects decoder-known
constant DC levels 0/0.5/1; it never receives an uncoded current-block mean.
Charge two mode bits per 8x8 block and a conservative fourteen motion-vector
bits per 16x16 P-frame block. This narrow virtual intra search is a declared
limitation, not an HEVC encoder implementation. Use separate I/P monotonic
factorized CDFs with independent parameters per RGB/DCT-frequency channel.
Fit once on 256 deterministic train clips (seed 9090), record the severity
schedule, source/runtime/data hashes, and freeze the same artifact for all jobs.
Use `10 * (MSE(decoded, original) + 0.001 * estimated_bpp) + task_loss`.
For the MOT17 adaptation task_loss is frozen-detector feature consistency;
original action-recognition/single-object-tracking analyzers are replaced.
Common severity 0..1 maps to the existing CRF-equivalent proxy range 22..47,
instead of the original paper's f_q range 30..50 and fixed-QP test points.

## Execution and evidence

Keep all changes additive; do not modify the active V3 evaluation or existing
checkpoints. Package new source with hashes in private Kaggle workers and pin
both upstream source revisions. Check fresh quota and existing workloads before
admission. Run a real-data GPU canary for each method before admitting all seeds.
Record data/source/teacher/checkpoint hashes, finite gradients and optimizer
updates, resume state, validation history and GPU/runtime identity. Preserve
incomplete runs and errors as such. Evaluate only eligible completed checkpoints.

Final outputs: raw detections and packet sizes, per-seed AP/bpp/BD-Rate,
mean/sample standard deviation, paired-method comparison and an assumptions
report. Labels must say **official-code SCOPE adapted to MOT17/x265** and
**independent Zhao reproduction adapted to MOT17/x265**. No result is presented
as a reproduction of the original papers' published COCO/GOT-10K numbers.
