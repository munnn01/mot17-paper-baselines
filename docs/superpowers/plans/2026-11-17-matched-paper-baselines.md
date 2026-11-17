# Matched paper baselines implementation plan

> Execute inline with superpowers:executing-plans. The user approved the design
> and instructed immediate execution with remaining details delegated on 2026-11-17.

**Goal:** Run official-code SCOPE and an independent Zhao implementation on MOT17
under identical external training/evaluation conditions.

**Architecture:** Add a separate baseline module, shared virtual codec, runner,
private Kaggle packager and result validator. Reuse the pinned project's actual
data loader, real x265 codec and detector/metric implementation without changing
the existing V3 jobs. Import hash-pinned upstream SCOPE network/loss helpers.

**Tech stack:** Python 3.11.14, PyTorch 2.5.1, torchvision 0.20.1, FFmpeg/x265,
Kaggle GPU API; local Python 3.12 CPU for contract tests.

**Spec:** `docs/superpowers/specs/2026-11-17-matched-paper-baselines-design.md`.

## Global constraints and review focus

All common conditions in the spec apply. No test data may influence training,
entropy calibration or checkpoint selection. A shared protocol hash must exclude
only method/seed identity and method-specific loss configuration. Incomplete
checkpoints must not be reported as final. Preserve existing jobs and files.
Reject source/data/teacher mismatch on resume and paired comparison. No secrets
in worker source, manifests or logs. Both workers must use the same GPU type.

## Tasks

- [x] Write behavioral tests for changed-frame temporal dependence, spatial
  frame independence, finite codec gradients, entropy calibration/freeze,
  round-trip transform, source tampering and matched protocol validation.
  Run `.runtime/eval-env/Scripts/python.exe -m pytest tests/test_paper_baselines.py`
  and observe failures before implementation.
- [x] Implement `models/paper_baselines.py`,
  `video_codecs/zhao_virtual_codec.py` and `scripts/run_paper_baselines.py`.
  Use tests above and the full suite. Keep upstream SCOPE source unchanged.
- [x] Implement a hash-verified standalone worker and local control in
  `scripts/kaggle_paper_baselines.py`. Generate private new kernel IDs, inspect
  quota/reservations, and avoid overwriting any kernel. Test embedded-source
  tampering rejection and actual generated worker bootstrap without network.
- [ ] Execute both real-MOT17 GPU canaries, inspect gradients/update/checkpoint
  evidence, then admit six independent production jobs. Track running status
  and resume interruptions without changing scientific budgets.
- [ ] Validate completed checkpoints and paired outputs; aggregate AP/bpp and
  BD-Rate with existing dataset-level metrics. Save status and assumptions
  report, distinguishing execution from final benchmark completion.

No new git commit is required for dispatch: workers carry the exact additive
source bytes and hashes, alongside two pinned upstream checkouts.

## Execution record

- Design approved; implemented inline. Fresh review found and regression-fixed
  free predictor side information, dependency drift, calibration provenance and
  detector/encoder identity gaps. Full suite before final export gate:
  final recheck: 180 passed, 2 skipped, 314 subtests. Compilation and diff checks
  also passed. This validates local code, not actual GPU memory/throughput.
- GPU stages remain unstarted: automatic approval rejected the attempted first
  calibration push because project-derived source export to Kaggle needs explicit
  user confirmation. No kernel was submitted. Ready-to-review worker and exact
  source/hash manifest are in `outputs_v3/paper_baselines_20261117/preview/`.
