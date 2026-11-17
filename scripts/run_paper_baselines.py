"""Separate, matched MOT17 paper adaptations. No V3 job is changed."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.paper_baselines import (OfficialScopeClip, ZhaoVideoPreprocessor,
    SCOPE_REVISION, feature_consistency, scope_objective, zhao_objective)
from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
from video_codecs.zhao_virtual_codec import ZhaoVirtualCodec
from video_codecs.hybrid_x265_ste import HybridX265STE
from scripts.experiment_protocol import CODEC_CONFIGURATION, METRIC_POLICY, CRFS, sha256_file
from scripts.run_v3 import (EpochDataset, load_real_dataset, data_fingerprint,
    execution_provenance, optimizer_step, require_teacher, save_checkpoint)

BASE_REVISION = "92d778c356331f5fb3d8f9e66f7388b33821a1ad"
METHODS = ("scope_official", "zhao_reproduction")
COMMON = {
    "schema_version": "paper-baselines-1", "img_size": 512, "clip_len": 4,
    "epochs": 36, "min_epochs": 12, "patience": 10, "min_delta": 0.0005,
    "batch_size": 2, "grad_accum_steps": 2, "num_workers": 2,
    "lr": 0.0001, "optimizer": "Adam", "weight_decay": 0.0,
    "max_hours": 7.8, "val_subsample": 2, "val_severities": [0.3, 0.7],
    "train_sequences": ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-10"],
    "val_sequences": ["MOT17-09"], "test_sequences": ["MOT17-11", "MOT17-13"],
    "train_stride": 2, "test_stride": 4, "crfs": CRFS,
    "teacher": "fasterrcnn_resnet50_fpn_v2", "train_yolo": False,
    "gpu": "Tesla T4", "precision": "float16_autocast_dynamic_scaler",
    "schedule": "warmup2_cosine20_floor1e-6", "gradient_clip": 1.0,
    "codec": CODEC_CONFIGURATION, "base_revision": BASE_REVISION,
    "scope_revision": SCOPE_REVISION,
    "calibration": {"seed": 9090, "clips": 256, "lr": 0.001,
                    "prior": "separate_IP_monotonic_CDF_RGB_8x8_frequency"},
}


def make_job(method, seed, entropy_sha256=None, source_sha256=None):
    if method not in METHODS or type(seed) is not int or seed not in (2026, 2027, 2028):
        raise ValueError("Unknown method or independent seed")
    return {**copy.deepcopy(COMMON), "method": method, "seed": seed,
            "job_id": f"{method}_seed{seed}", "entropy_sha256": entropy_sha256,
            "source_sha256": copy.deepcopy(source_sha256 or {})}


def validate_job(job):
    expected = make_job(job.get("method"), job.get("seed"),
                        job.get("entropy_sha256"), job.get("source_sha256"))
    if job != expected:
        raise ValueError("Job differs from the matched experiment protocol")
    def digest(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if job["entropy_sha256"] is not None and not digest(job["entropy_sha256"]):
        raise ValueError("Invalid entropy artifact hash")
    sources = job["source_sha256"]
    if sources and (set(sources) != {"models/paper_baselines.py", "video_codecs/zhao_virtual_codec.py", "scripts/run_paper_baselines.py"}
                    or not all(digest(value) for value in sources.values())):
        raise ValueError("Unexpected source paths or hashes")
    return job


def protocol_hash(job):
    validate_job(job)
    common = {k: v for k, v in job.items() if k not in ("method", "seed", "job_id")}
    return hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(v) for v in value)
    return True


def validate_final_checkpoint(state, job):
    if state.get("job") != job:
        raise ValueError("Checkpoint belongs to a different protocol/job")
    if state.get("status") not in ("completed", "converged") or not state.get("best_state"):
        raise ValueError("Final evaluation requires completed or converged selected weights")
    if not finite_tensors(state):
        raise ValueError("Checkpoint tensors must be finite")
    history = state.get("history")
    if not isinstance(history, list) or not history or [r.get("epoch") for r in history] != list(range(1, len(history)+1)):
        raise ValueError("Completed checkpoint requires contiguous validation history")
    if any(type(r.get("val_loss")) not in (int, float) or not math.isfinite(r["val_loss"]) for r in history):
        raise ValueError("Validation history must be finite")
    best = state.get("best_epoch", 0)
    if type(best) is not int or not 1 <= best <= len(history) or not math.isclose(
            state.get("best_loss", float("nan")), history[best-1]["val_loss"], abs_tol=1e-7):
        raise ValueError("Best weights do not match validation history")
    if state["status"] == "completed" and len(history) != 36:
        raise ValueError("Completed checkpoint has not reached the epoch ceiling")
    if state["status"] == "converged" and (len(history) < 12 or len(history)-best < 10):
        raise ValueError("Converged checkpoint has not met minimum epochs/patience")
    if state.get("epoch") != len(history)+1 or state.get("next_batch") != 0 or state.get("optimizer_updates", 0) <= 0:
        raise ValueError("Checkpoint position does not represent final completed training")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def runtime_proof(job):
    proof = execution_provenance()
    proof["source_sha256"].update({n: sha256_file(ROOT / n) for n in job["source_sha256"]})
    if proof["gpu"] != job["gpu"]:
        raise RuntimeError("Matched experiment requires the selected Tesla T4 GPU")
    for name, digest in job["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Packaged source hash mismatch")
    return proof


def calibration_provenance():
    proof = execution_provenance()
    return {"codec_sha256": sha256_file(ROOT / "video_codecs/zhao_virtual_codec.py"),
            "motion_source_sha256": sha256_file(ROOT / "video_codecs/hybrid_x265_ste.py"),
            "runtime": {key: proof[key] for key in
                ("python", "torch", "torchvision", "numpy", "cuda", "gpu")}}


def validate_calibration_provenance(proof):
    if proof != calibration_provenance():
        raise ValueError("Shared calibration provenance differs from codec source/runtime")


def detector_identity(teacher, include_yolo=False):
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
    filename = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.url.rsplit("/", 1)[-1]
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    result = {"fasterrcnn_sha256": sha256_file(path)}
    if include_yolo:
        wrapper = teacher._yolo_wrapper_holder[0]
        path = Path(getattr(wrapper, "ckpt_path", "yolov8s.pt"))
        result["yolov8s_sha256"] = sha256_file(path)
        import ultralytics
        result["ultralytics"] = ultralytics.__version__
    return result


def encoder_identity():
    version = subprocess.check_output(["ffmpeg", "-version"], text=True)
    probe = subprocess.check_output(["ffprobe", "-version"], text=True)
    check = subprocess.run(["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "color=size=32x32:rate=1",
        "-frames:v", "1", "-c:v", "libx265", "-x265-params", "pools=1:log-level=info",
        "-f", "null", "-"], text=True, capture_output=True, check=True)
    x265 = [line for line in check.stderr.splitlines() if "HEVC encoder version" in line or "x265 [info]: build info" in line]
    if not x265:
        raise RuntimeError("Actual libx265 build identity is unavailable")
    return {"ffmpeg": version.splitlines()[0], "ffmpeg_build_sha256": hashlib.sha256(version.encode()).hexdigest(),
            "ffprobe_build_sha256": hashlib.sha256(probe.encode()).hexdigest(), "libx265": x265}


def calibrate(data_root, output):
    """One training-only artifact, attached version-pinned to every later job."""
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "Tesla T4":
        raise RuntimeError("Calibration requires matching Tesla T4 GPU")
    torch.manual_seed(9090)
    torch.set_num_threads(2)
    dataset = load_real_dataset(COMMON, data_root, "train")
    codec = ZhaoVirtualCodec().cuda()
    optimizer = torch.optim.Adam(codec.entropy.parameters(), lr=0.001)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(9090))[:256].tolist()
    samples = EpochDataset(dataset, 9090, 0)
    records, severities = [], []
    for step, index in enumerate(indices):
        video = samples[index]["video"][None].cuda()
        severity = torch.rand(1, generator=torch.Generator().manual_seed(9090 + step)).cuda()
        severities.append(severity.item())
        with torch.no_grad():
            values = codec(video, severity, return_symbols=True)["calibration_symbols"]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack([-codec.entropy[i](symbols).log2().mean() for i, symbols in values]).mean()
        loss.backward()
        if not torch.isfinite(loss) or not finite_tensors([p.grad for p in codec.entropy.parameters()]):
            raise RuntimeError("Nonfinite entropy calibration")
        optimizer.step()
        records.append(float(loss))
        if (step + 1) % 16 == 0:
            print(f"PAPER_CALIBRATION clips={step+1}/256 nll={loss.item():.6f}", flush=True)
    path = Path(output) / "shared_entropy.pt"
    state = {"entropy_state": codec.entropy.state_dict(), "configuration": COMMON["calibration"],
             "training_data": data_fingerprint(dataset), "clip_indices": indices, "losses": records,
             "severities": severities, "provenance": calibration_provenance()}
    save_checkpoint(path, state)
    proof = {"artifact_sha256": sha256_file(path), "clips": len(indices),
             "training_data": state["training_data"], "status": "calibrated"}
    write_json(Path(output) / "shared_entropy.json", proof)
    print("PAPER_CALIBRATED " + json.dumps(proof), flush=True)


def load_codec(job, path, device):
    if not job["entropy_sha256"] or sha256_file(path) != job["entropy_sha256"]:
        raise ValueError("Shared entropy artifact hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    validate_calibration_provenance(state.get("provenance"))
    if state["configuration"] != COMMON["calibration"] or len(state["clip_indices"]) != 256 or not finite_tensors(state):
        raise ValueError("Invalid calibrated prior")
    codec = ZhaoVirtualCodec().to(device)
    codec.entropy.load_state_dict(state["entropy_state"], strict=True)
    return codec.freeze_entropy(), state["training_data"]


def build_model(job, scope_root):
    return OfficialScopeClip(scope_root) if job["method"] == "scope_official" else ZhaoVideoPreprocessor()


def batch_loss(model, codec, teacher, batch, job, device, severity):
    video = batch["video"].to(device)
    out = model(video, severity)
    decoded = codec(out["edited"], severity)
    if job["method"] == "scope_official":
        return scope_objective(model, out, video, decoded["video_recon"], decoded["rate_bpp"],
                               batch["m_attn"].to(device), teacher)
    task = feature_consistency(teacher, video.flatten(0, 1), decoded["video_recon"].flatten(0, 1))
    return zhao_objective(video, decoded["video_recon"], decoded["rate_bpp"], task)


@torch.no_grad()
def validation(model, codec, teacher, dataset, job, device, deadline):
    model.eval()
    total = count = 0
    loader = DataLoader(Subset(dataset, range(0, len(dataset), 2)), batch_size=2, num_workers=0)
    for batch in loader:
        n = batch["video"].shape[0]
        for severity in (0.3, 0.7):
            if time.monotonic() >= deadline:
                raise TimeoutError("Session interrupted during validation")
            with torch.amp.autocast("cuda"):
                value = batch_loss(model, codec, teacher, batch, job, device,
                                   torch.full((n,), severity, device=device))
            if not torch.isfinite(value):
                raise RuntimeError("Nonfinite validation loss")
            total += value.item() * n
            count += n
    return total / count


def train(job, data_root, output, scope_root, entropy_path, resume=None, canary=False):
    validate_job(job)
    device = torch.device("cuda")
    torch.set_num_threads(2)
    proof = runtime_proof(job)
    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])
    dataset = load_real_dataset(job, data_root, "train")
    val = load_real_dataset(job, data_root, "val")
    proof["training_data"], proof["validation_data"] = data_fingerprint(dataset), data_fingerprint(val)
    codec, calibration_data = load_codec(job, entropy_path, device)
    if proof["training_data"] != calibration_data:
        raise ValueError("Calibration used a different training dataset")
    model = build_model(job, scope_root).to(device)
    teacher = FrozenFPNDetectorTeacher(pretrained=True).to(device).eval()
    require_teacher(teacher, training=True)
    proof["teacher_sha256"] = detector_identity(teacher)["fasterrcnn_sha256"]
    optimizer = torch.optim.Adam(model.parameters(), lr=job["lr"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (job["job_id"] + (".canary.pt" if canary else ".last.pt"))
    if path.exists():
        raise FileExistsError("Use a new output path or an explicit new resume session")
    epoch, next_batch, updates, overflows, best_epoch = 1, 0, 0, 0, 0
    best_loss, best_state, history = float("inf"), None, []
    epoch_sum, epoch_count = 0.0, 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["job"] != job or state["status"] != "time_budget_exhausted" or not finite_tensors(state):
            raise ValueError("Invalid incomplete resume state")
        for k in ("training_data", "validation_data", "teacher_sha256", "source_sha256", "gpu"):
            if state["provenance"][k] != proof[k]:
                raise ValueError("Resume data/teacher/source/runtime differs")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        epoch, next_batch, updates = state["epoch"], state["next_batch"], state["optimizer_updates"]
        best_loss, best_epoch, best_state = state["best_loss"], state["best_epoch"], state["best_state"]
        history, epoch_sum, epoch_count = state["history"], state["epoch_sum"], state["epoch_count"]
    started = time.monotonic()
    deadline = started + job["max_hours"] * 3600
    status = "running"
    def snapshot():
        save_checkpoint(path, {"job": job, "status": status, "epoch": epoch,
            "next_batch": next_batch, "optimizer_updates": updates,
            "amp_skipped_updates": overflows, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
            "best_loss": best_loss, "best_epoch": best_epoch, "best_state": best_state,
            "history": history, "epoch_sum": epoch_sum, "epoch_count": epoch_count,
            "provenance": proof, "protocol_sha256": protocol_hash(job)})
    while epoch <= job["epochs"]:
        loader = DataLoader(EpochDataset(dataset, job["seed"], epoch), batch_size=2,
            shuffle=True, num_workers=2, pin_memory=True,
            generator=torch.Generator().manual_seed(job["seed"] + epoch))
        warmup = min(1.0, epoch / 2.0)
        progress = min(1.0, max(0.0, (epoch - 2) / 18.0))
        lr = warmup * (1e-6 + (job["lr"] - 1e-6) * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            if index < next_batch:
                continue
            rng = torch.Generator().manual_seed(job["seed"] + epoch * 1000003 + index)
            severity = torch.rand(batch["video"].shape[0], generator=rng).to(device)
            window = min(2, len(loader) - (index // 2) * 2)
            with torch.amp.autocast("cuda"):
                loss = batch_loss(model, codec, teacher, batch, job, device, severity)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training objective")
            scaler.scale(loss / window).backward()
            epoch_sum += loss.item()
            epoch_count += 1
            next_batch = index + 1
            if next_batch % 2 == 0 or next_batch == len(loader):
                step = optimizer_step(model, optimizer, scaler)
                if step["updated"]:
                    updates += 1
                    overflows = 0
                    print(f"PAPER_STEP {job['job_id']} epoch={epoch} batch={next_batch}/{len(loader)} update={updates} loss={loss.item():.6f} grad_norm={step['grad_norm']:.6f}", flush=True)
                else:
                    overflows += 1
                    if overflows >= 16:
                        raise RuntimeError("Persistent AMP overflow")
                if (canary and updates >= 128) or time.monotonic() >= deadline:
                    status = "canary_passed" if canary and updates >= 128 else "time_budget_exhausted"
                    snapshot()
                    result = {"status": status, "optimizer_updates": updates, "job_id": job["job_id"],
                        "seconds": time.monotonic()-started, "checkpoint_sha256": sha256_file(path),
                        "protocol_sha256": protocol_hash(job), "gpu": proof["gpu"],
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
                    restored = torch.load(path, map_location="cpu", weights_only=True)
                    if not finite_tensors(restored) or restored["optimizer_updates"] != updates:
                        raise RuntimeError("Checkpoint round-trip failed")
                    write_json(out / (job["job_id"] + ".result.json"), result)
                    print("PAPER_RESULT " + json.dumps(result), flush=True)
                    return result
                if updates <= 2 or updates % 40 == 0:
                    snapshot()
        snapshot()
        try:
            value = validation(model, codec, teacher, val, job, device, deadline)
        except TimeoutError:
            status = "time_budget_exhausted"
            snapshot()
            return {"status": status}
        history.append({"epoch": epoch, "val_loss": value, "train_loss": epoch_sum/epoch_count, "lr": lr})
        if value < best_loss - job["min_delta"]:
            best_loss, best_epoch = value, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        finished_epoch = epoch
        epoch, next_batch, epoch_sum, epoch_count = epoch + 1, 0, 0.0, 0
        if finished_epoch >= 12 and finished_epoch - best_epoch >= 10:
            status = "converged"
        elif finished_epoch == 36:
            status = "completed"
        snapshot()
        print("PAPER_EPOCH " + json.dumps(history[-1]), flush=True)
        if status != "running":
            write_json(out / (job["job_id"] + ".result.json"), {"status": status, "optimizer_updates": updates})
            return {"status": status}


@torch.no_grad()
def evaluate(job, checkpoint, data_root, output, scope_root):
    validate_job(job)
    proof = runtime_proof(job)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_final_checkpoint(state, job)
    dataset = load_real_dataset(job, data_root, "test")
    model = build_model(job, scope_root).cuda().eval()
    model.load_state_dict(state["best_state"], strict=True)
    teacher = FrozenFPNDetectorTeacher(pretrained=True, use_yolov8s=True).cuda().eval()
    require_teacher(teacher, require_yolo=True)
    proof["detectors"] = detector_identity(teacher, include_yolo=True)
    if proof["detectors"]["fasterrcnn_sha256"] != state["provenance"]["teacher_sha256"]:
        raise ValueError("Evaluation teacher differs from training weights")
    proof["encoder"] = encoder_identity()
    codec = HybridX265STE(use_real_x265_in_forward=True).cuda()
    out = Path(output)
    record_path = out / (job["job_id"] + ".records.jsonl")
    with record_path.open("x", encoding="utf-8") as handle:
        for index in range(len(dataset)):
            sample = dataset[index]
            video = sample["video"][None].cuda()
            for crf in CRFS:
                with torch.amp.autocast("cuda"):
                    edited = model(video, torch.tensor([(crf-22)/25.0], device="cuda"))["edited"]
                for method, frames in (("anchor_x265", video), ("model", edited)):
                    decoded = codec.evaluate_real_x265(frames, crf, None)
                    metrics = teacher.evaluate_task_accuracy(video, decoded["video_recon"], sample["m_det"][None].cuda(),
                        gt_boxes=sample["gt_boxes"][None].cuda(), ignore_boxes=sample["ignore_boxes"][None].cuda(),
                        require_real_detector=True, return_raw_detections=True)
                    raw = {k: [v.tolist() for v in values] for k, values in metrics["raw_detections"].items()
                           if k in ("gt_boxes", "ignore_boxes", "pred_boxes", "pred_scores", "yolo_pred_boxes", "yolo_pred_scores")}
                    record = {"method": method, "crf": crf, "frame_ids": sample["frame_ids"],
                        "packet_sizes": decoded["packet_sizes"], "frame_types": decoded["frame_types"],
                        "height": 512, "width": 512, "raw_detections": raw,
                        "task_roi_psnr_db": metrics["task_roi_psnr_db"]}
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            print(f"PAPER_EVAL {job['job_id']} clip={index+1}/{len(dataset)}", flush=True)
    metadata = {"job": job, "checkpoint_sha256": sha256_file(checkpoint),
        "records_sha256": sha256_file(record_path), "records_file": record_path.name,
        "test_data": data_fingerprint(dataset), "protocol_sha256": protocol_hash(job),
        "metric_policy": METRIC_POLICY, "provenance": proof, "checkpoint_status": state["status"],
        "expected_frame_ids": dataset.physical_frame_manifest(), "evaluation_yolo": True}
    write_json(out / (job["job_id"] + ".eval.json"), metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "canary", "train", "eval"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-file")
    parser.add_argument("--scope-root", default=str(ROOT / ".runtime/SCOPE"))
    parser.add_argument("--entropy")
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.mode == "calibrate":
        return calibrate(args.data_root, args.output_dir)
    job = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
    if args.mode == "eval":
        return evaluate(job, args.checkpoint, args.data_root, args.output_dir, args.scope_root)
    return train(job, args.data_root, args.output_dir, args.scope_root, args.entropy,
                 resume=args.resume, canary=args.mode == "canary")


if __name__ == "__main__":
    main()
