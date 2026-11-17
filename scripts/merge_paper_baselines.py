"""Strict paired MOT17 metrics for the six independently trained paper jobs."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_paper_baselines import METHODS, validate_job, protocol_hash
from scripts.experiment_protocol import EXPECTED_TEST_FRAME_IDS, METRIC_POLICY, CRFS, sha256_file
from scripts.merge_v3 import _validated_records, _comparisons, _fingerprint
from scripts.merge_shards import summarize_records


def strict_json(text):
    def reject(value):
        raise ValueError("Nonfinite JSON measurement")
    return json.loads(text, parse_constant=reject)


def stats(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None}


def merge_evaluations(metadata_paths, allow_partial=False):
    if not metadata_paths and not allow_partial:
        raise ValueError("Final report requires all six independent paper jobs")
    expected = {f"{method}_seed{seed}" for method in METHODS for seed in (2026, 2027, 2028)}
    jobs, common, ground_truth, anchor = {}, None, {}, None
    for item in metadata_paths:
        path = Path(item)
        metadata = strict_json(path.read_text(encoding="utf-8"))
        job = validate_job(metadata["job"])
        if not job["entropy_sha256"] or not job["source_sha256"]:
            raise ValueError("Final evaluation requires pinned entropy and source hashes")
        name = job["job_id"]
        if name in jobs or name not in expected:
            raise ValueError("Duplicate or unexpected independent paper job")
        if metadata.get("checkpoint_status") not in ("completed", "converged"):
            raise ValueError("Incomplete checkpoint cannot enter final benchmark")
        if metadata.get("protocol_sha256") != protocol_hash(job):
            raise ValueError("Evaluation protocol hash differs")
        if metadata.get("metric_policy") != METRIC_POLICY or metadata.get("evaluation_yolo") is not True:
            raise ValueError("Metric/detector policy differs from matched evaluation")
        if metadata.get("expected_frame_ids") != EXPECTED_TEST_FRAME_IDS:
            raise ValueError("Held-out physical frame manifest differs")
        _fingerprint(metadata.get("test_data"))
        proof = metadata["provenance"]
        for key in ("fasterrcnn_sha256", "yolov8s_sha256"):
            digest = proof.get("detectors", {}).get(key)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("Missing detector weight hash")
        if not proof.get("encoder", {}).get("libx265"):
            raise ValueError("Missing actual encoder build identity")
        cohort = {"protocol": protocol_hash(job), "data": metadata["test_data"],
            "runtime": {k: v for k, v in proof.items() if k != "measured_at_utc"}}
        if common is None:
            common = cohort
        elif common != cohort:
            raise ValueError("Jobs differ in data, sources, detectors, encoder or runtime")
        filename = name + ".records.jsonl"
        records = path.parent / filename
        if metadata.get("records_file") != filename or records.is_symlink() or sha256_file(records) != metadata["records_sha256"]:
            raise ValueError("Evaluation records hash/path mismatch")
        groups = _validated_records(metadata, records, ground_truth)
        methods = {method: {str(crf): summarize_records(groups[method, crf]["records"])
                            for crf in CRFS} for method in ("anchor_x265", "model")}
        if anchor is None:
            anchor = methods["anchor_x265"]
        elif anchor != methods["anchor_x265"]:
            raise ValueError("Anchor curves differ across supposedly paired jobs")
        jobs[name] = {"method": job["method"], "seed": job["seed"], "curves": methods,
            "bd_rate": _comparisons(methods["model"], methods["anchor_x265"], True),
            "checkpoint_sha256": metadata["checkpoint_sha256"]}
    missing = sorted(expected - set(jobs))
    if missing and not allow_partial:
        raise ValueError("Final report requires all six independent paper jobs: " + ", ".join(missing))
    aggregate = {}
    for method in METHODS:
        rows = [r for r in jobs.values() if r["method"] == method]
        fields = {}
        names = rows[0]["bd_rate"] if rows else []
        for name in names:
            values = [r["bd_rate"][name]["primary"]["value_pct"] for r in rows
                      if r["bd_rate"][name]["primary"] is not None]
            fields[name] = {**stats(values), "expected_n": 3,
                "unavailable_seeds": [r["seed"] for r in rows if r["bd_rate"][name]["primary"] is None]}
        aggregate[method] = fields
    paired = {}
    for seed in (2026, 2027, 2028):
        scope, zhao = (jobs.get(f"{m}_seed{seed}") for m in METHODS)
        if scope and zhao:
            paired[str(seed)] = _comparisons(zhao["curves"]["model"], scope["curves"]["model"], True)
    return {"schema": "matched-paper-baselines-1", "final_ready": not missing,
        "missing_jobs": missing, "conditions": common, "jobs": jobs,
        "aggregate_bd_rate": aggregate, "zhao_vs_scope_paired": paired,
        "labels": {"scope_official": "Official-code SCOPE adapted to MOT17/x265",
                   "zhao_reproduction": "Independent Zhao reproduction adapted to MOT17/x265"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = merge_evaluations(sorted(Path(args.input_dir).rglob("*.eval.json")), args.allow_partial)
    with Path(args.output).open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
