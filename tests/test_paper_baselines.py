"""Behavioral contracts for the independent, matched paper baselines."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def feature(name):
    assert importlib.util.find_spec(name) is not None, f"Missing baseline component: {name}"
    return importlib.import_module(name)


def test_zhao_uses_other_frames_and_preserves_rgb_shape():
    m = feature("models.paper_baselines")
    torch.manual_seed(7)
    model = m.ZhaoVideoPreprocessor().eval()
    video = torch.full((1, 4, 3, 32, 32), 0.4)
    changed = video.clone()
    changed[:, 0] = 0.9
    a, b = model(video)["edited"], model(changed)["edited"]
    assert a.shape == video.shape
    assert not torch.allclose(a[:, 2], b[:, 2], atol=1e-7)
    a.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is not None for p in model.parameters())


def test_official_scope_is_framewise_and_rejects_modified_source(tmp_path):
    m = feature("models.paper_baselines")
    upstream = ROOT / ".runtime/SCOPE"
    model = m.OfficialScopeClip(upstream).eval()
    x = torch.full((1, 4, 3, 32, 32), 0.4)
    y = x.clone()
    y[:, 0] = 0.8
    with torch.no_grad():
        a = model(x, torch.tensor([0.5]))["edited"]
        b = model(y, torch.tensor([0.5]))["edited"]
    assert torch.equal(a[:, 2], b[:, 2])
    (tmp_path / "scope").mkdir()
    for name in ("model.py", "losses.py"):
        (tmp_path / "scope" / name).write_bytes((upstream / "scope" / name).read_bytes())
    with (tmp_path / "scope/model.py").open("ab") as handle:
        handle.write(b"\n# tampered\n")
    with pytest.raises(ValueError, match="hash"):
        m.OfficialScopeClip(tmp_path)


def test_codec_roundtrip_finite_gradients_and_increasing_quantization():
    m = feature("video_codecs.zhao_virtual_codec")
    torch.manual_seed(11)
    codec = m.ZhaoVirtualCodec().freeze_entropy()
    x = torch.rand(1, 4, 3, 32, 32, requires_grad=True)
    mild = codec(x, torch.tensor([0.0]))
    severe = codec(x, torch.tensor([1.0]))
    assert mild["video_recon"].shape == x.shape
    assert severe["q_step"].item() > mild["q_step"].item()
    assert torch.isfinite(mild["rate_bpp"]) and mild["rate_bpp"] > 0
    (mild["rate_bpp"] + mild["video_recon"].square().mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert not any(p.requires_grad for p in codec.entropy.parameters())
    image = torch.rand(1, 3, 32, 32)
    assert torch.allclose(codec.idct(codec.dct(image), 32, 32), image, atol=2e-6)


def test_factorized_cdf_calibration_changes_density_then_freezes():
    m = feature("video_codecs.zhao_virtual_codec")
    torch.manual_seed(5)
    prior = m.FactorizedCDF(3)
    symbols = torch.randn(3, 1, 100)
    before = prior(symbols).detach().clone()
    optimizer = torch.optim.Adam(prior.parameters(), lr=0.01)
    (-prior(symbols).log2().mean()).backward()
    optimizer.step()
    assert not torch.equal(before, prior(symbols).detach())
    samples = torch.tensor([-10., 0., 10.]).reshape(1, 1, 3).expand(3, 1, 3)
    cdf = prior.cdf(samples)
    assert torch.all(cdf[..., 1:] > cdf[..., :-1])


def test_matched_protocol_rejects_different_data_codec_and_budget():
    m = feature("scripts.run_paper_baselines")
    a = m.make_job("scope_official", 2026)
    b = m.make_job("zhao_reproduction", 2027)
    assert m.protocol_hash(a) == m.protocol_hash(b)
    for key, value in (("img_size", 256), ("epochs", 12), ("lr", 0.0002)):
        altered = copy.deepcopy(b)
        altered[key] = value
        with pytest.raises(ValueError):
            m.validate_job(altered)
    altered = copy.deepcopy(b)
    altered["codec"]["preset"] = "fast"
    with pytest.raises(ValueError):
        m.validate_job(altered)


def test_zhao_objective_keeps_full_frame_mse_and_paper_weights():
    m = feature("models.paper_baselines")
    clean = torch.zeros(1, 4, 3, 8, 8)
    decoded = torch.ones_like(clean) * 0.5
    value = m.zhao_objective(clean, decoded, torch.tensor(2.), torch.tensor(3.))
    assert value.item() == pytest.approx(5.52)


def test_checkpoint_rejects_nonfinite_incomplete_or_other_protocol():
    m = feature("scripts.run_paper_baselines")
    job = m.make_job("zhao_reproduction", 2026)
    state = {"job": job, "status": "time_budget_exhausted", "best_state": {"w": torch.ones(2)}}
    with pytest.raises(ValueError, match="completed"):
        m.validate_final_checkpoint(state, job)
    state["status"] = "completed"
    state["best_state"]["w"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        m.validate_final_checkpoint(state, job)


def test_packaged_source_is_verified_before_any_file_is_written(tmp_path):
    m = feature("scripts.kaggle_paper_baselines")
    payload = m.pack_sources(ROOT)
    m.unpack_sources(payload, tmp_path)
    assert (tmp_path / "models/paper_baselines.py").read_bytes() == (ROOT / "models/paper_baselines.py").read_bytes()
    altered = copy.deepcopy(payload)
    altered["hashes"]["models/paper_baselines.py"] = "0" * 64
    other = tmp_path / "invalid"
    with pytest.raises(ValueError, match="hash"):
        m.unpack_sources(altered, other)
    assert not other.exists()


def test_generated_worker_contains_only_allowlisted_source_and_compiles():
    m = feature("scripts.kaggle_paper_baselines")
    worker = m.build_worker({"mode": "calibrate"}, ROOT)
    compile(worker, "worker.py", "exec")
    payload = m.pack_sources(ROOT)
    assert set(payload["hashes"]) == {"models/paper_baselines.py", "video_codecs/zhao_virtual_codec.py", "scripts/run_paper_baselines.py"}
    assert "KGAT_" not in worker


def test_codec_does_not_receive_current_block_mean_as_free_side_information():
    m = feature("video_codecs.zhao_virtual_codec")
    codec = m.ZhaoVirtualCodec().freeze_entropy()
    x = torch.full((1, 1, 3, 32, 32), 0.4137)
    output = codec(x, torch.tensor([0.7]))
    # Reconstructing this arbitrary constant perfectly would reveal an
    # unquantized current-block mean supplied to the decoder for free.
    assert (output["video_recon"] - x).abs().max().item() > 1e-5


def test_evaluation_dependency_install_cannot_upgrade_common_torch_stack():
    m = feature("scripts.kaggle_paper_baselines")
    # Executing the bootstrap itself requires network/GPU; its generated
    # installer argv is tested through the helper used by the bootstrap.
    command = m.evaluation_dependencies("/tmp/python", "/tmp/repo")
    assert command[-2:] == ["--constraint", str(Path("/tmp/repo") / "requirements.txt")]


def test_calibration_rejects_prior_fitted_with_another_codec_source():
    m = feature("scripts.run_paper_baselines")
    proof = {"codec_sha256": "0" * 64, "runtime": {"gpu": "Tesla T4"}}
    with pytest.raises(ValueError, match="calibration provenance"):
        m.validate_calibration_provenance(proof)


def test_merge_requires_a_complete_paired_matrix():
    m = feature("scripts.merge_paper_baselines")
    with pytest.raises(ValueError, match="six"):
        m.merge_evaluations([])


def test_production_admission_requires_two_verified_real_data_canaries():
    m = feature("scripts.kaggle_paper_baselines")
    row = {"config": {"mode": "train"}}
    with pytest.raises(ValueError, match="canaries"):
        m.admission_check({"jobs": []}, row)


def test_export_rejection_remains_blocked_until_explicit_approval(tmp_path):
    m = feature("scripts.kaggle_paper_baselines")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"export_approved": False, "jobs": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="export approval"):
        m.operate(path, "push")


def test_removed_notebook_does_not_block_current_workload_inspection():
    from types import SimpleNamespace
    m = feature("scripts.kaggle_paper_baselines")
    class Removed(Exception):
        response = SimpleNamespace(status_code=404)
    class Api:
        def kernels_list_with_response(self, **kwargs):
            return SimpleNamespace(kernels=[SimpleNamespace(ref="owner/removed"),
                SimpleNamespace(ref="owner/live")], next_page_token=None)
        def kernels_status(self, ref):
            if ref.endswith("removed"):
                raise Removed()
            return SimpleNamespace(status="running")
    assert m.active_workloads(Api(), "owner") == ["owner/live"]


def test_published_launcher_reads_the_explicit_private_pool_path(tmp_path, monkeypatch):
    m = feature("scripts.kaggle_paper_baselines")
    api = importlib.import_module("scripts.kaggle_v3")
    received = []
    monkeypatch.setattr(api, "load_account_pool", lambda path: received.append(path) or [])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"jobs": []}), encoding="utf-8")
    m.operate(manifest, "status", pool_file="/private/credentials.json")
    assert received == ["/private/credentials.json"]


def test_checkpoint_status_cannot_fake_completed_training():
    m = feature("scripts.run_paper_baselines")
    job = m.make_job("zhao_reproduction", 2026)
    state = {"job": job, "status": "completed", "best_state": {"w": torch.ones(2)}}
    with pytest.raises(ValueError, match="history"):
        m.validate_final_checkpoint(state, job)


def test_merge_measures_rate_savings_and_rejects_unpaired_frames(tmp_path, monkeypatch):
    m = feature("scripts.merge_paper_baselines")
    runner = feature("scripts.run_paper_baselines")
    v3 = importlib.import_module("scripts.merge_v3")
    ids = [f"MOT17-11/{i:06d}" for i in range(1, 9)]
    monkeypatch.setattr(m, "EXPECTED_TEST_FRAME_IDS", ids)
    monkeypatch.setattr(v3, "EXPECTED_TEST_FRAME_IDS", ids)
    paths = []
    for method, scale in (("scope_official", 0.8), ("zhao_reproduction", 0.7)):
        for seed in (2026, 2027, 2028):
            job = runner.make_job(method, seed, entropy_sha256="f"*64,
                source_sha256={name: "1"*64 for name in ("models/paper_baselines.py",
                    "video_codecs/zhao_virtual_codec.py", "scripts/run_paper_baselines.py")})
            rows = []
            for k, crf in enumerate((22, 27, 32, 37)):
                for label in ("anchor_x265", "model"):
                    packet = round((1000, 500, 250, 100)[k] * (scale if label == "model" else 1))
                    for clip in range(2):
                        raw = {"gt_boxes": [[[0, 0, 10, 10]]] * 4, "ignore_boxes": [[]]*4,
                               "pred_boxes": [[ [0, 0, 10, 10] ] if i < 4-k else [] for i in range(4)],
                               "pred_scores": [[0.9] if i < 4-k else [] for i in range(4)]}
                        raw["yolo_pred_boxes"] = raw["pred_boxes"]
                        raw["yolo_pred_scores"] = raw["pred_scores"]
                        rows.append({"method": label, "crf": crf, "frame_ids": ids[clip*4:clip*4+4],
                            "packet_sizes": [packet]*4, "frame_types": ["I", "P", "P", "P"],
                            "height": 512, "width": 512, "raw_detections": raw})
            record = tmp_path / (job["job_id"]+".records.jsonl")
            record.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
            metadata = {"job": job, "checkpoint_status": "completed", "checkpoint_sha256": "a"*64,
                "protocol_sha256": runner.protocol_hash(job), "metric_policy": m.METRIC_POLICY,
                "evaluation_yolo": True, "expected_frame_ids": ids, "records_file": record.name,
                "records_sha256": m.sha256_file(record), "test_data": {"physical_frames": 8,
                    "images_sha256": "b"*64, "gt_sha256": {"MOT17-11": "c"*64}},
                "provenance": {"gpu": "Tesla T4", "detectors": {"fasterrcnn_sha256": "d"*64,
                    "yolov8s_sha256": "e"*64}, "encoder": {"libx265": ["fixture-build"]}}}
            path = tmp_path / (job["job_id"]+".eval.json")
            path.write_text(json.dumps(metadata), encoding="utf-8")
            paths.append(path)
    result = m.merge_evaluations(paths)
    assert result["final_ready"]
    scope = result["aggregate_bd_rate"]["scope_official"]["bdbr_tot_gt_ap50_pct"]
    assert scope["mean"] == pytest.approx(-20)
    assert scope["n"] == 3 and scope["std"] == pytest.approx(0)
    assert result["zhao_vs_scope_paired"]["2026"]["bdbr_tot_gt_ap50_pct"]["primary"]["value_pct"] == pytest.approx(-12.5)
    metadata = json.loads(paths[0].read_text())
    original = copy.deepcopy(metadata)
    metadata["job"]["entropy_sha256"] = None
    metadata["protocol_sha256"] = runner.protocol_hash(metadata["job"])
    paths[0].write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="entropy"):
        m.merge_evaluations(paths)
    metadata = original
    paths[0].write_text(json.dumps(metadata), encoding="utf-8")
    record = tmp_path / metadata["records_file"]
    rows = [json.loads(line) for line in record.read_text().splitlines()]
    rows.pop()
    record.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    metadata["records_sha256"] = m.sha256_file(record)
    paths[0].write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        m.merge_evaluations(paths)
