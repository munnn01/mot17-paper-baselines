"""Private, source-verified Kaggle execution for matched paper baselines."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_FILES = ("models/paper_baselines.py", "video_codecs/zhao_virtual_codec.py", "scripts/run_paper_baselines.py")


def pack_sources(root):
    files = {name: base64.b64encode((Path(root)/name).read_bytes()).decode() for name in SOURCE_FILES}
    hashes = {name: hashlib.sha256(base64.b64decode(data)).hexdigest() for name, data in files.items()}
    return {"files": files, "hashes": hashes}


def unpack_sources(payload, destination):
    """Validate the whole allowlisted bundle before writing any files."""
    import base64
    import hashlib
    from pathlib import Path
    allowed = {"models/paper_baselines.py", "video_codecs/zhao_virtual_codec.py", "scripts/run_paper_baselines.py"}
    if set(payload["files"]) != allowed or set(payload["hashes"]) != allowed:
        raise ValueError("Source bundle contains unexpected paths")
    files = {name: base64.b64decode(data, validate=True) for name, data in payload["files"].items()}
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != payload["hashes"][name]:
            raise ValueError("Source bundle hash mismatch")
    destination = Path(destination)
    for name, data in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def evaluation_dependencies(python, repo):
    from pathlib import Path
    return ["ultralytics==8.3.230", "--constraint", str(Path(repo) / "requirements.txt")]


def build_worker(config, root=ROOT):
    from scripts.run_paper_baselines import BASE_REVISION
    from models.paper_baselines import SCOPE_REVISION
    config = dict(config, base_revision=BASE_REVISION, scope_revision=SCOPE_REVISION)
    payload = pack_sources(root)
    config["source_sha256"] = payload["hashes"]
    if config.get("job"):
        config["job"] = dict(config["job"], source_sha256=payload["hashes"])
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
    return '''"""Source-verified private paper baseline; no launcher credentials."""
import base64, hashlib, json, os, subprocess, sys, zlib
from pathlib import Path
CONFIG = json.loads(''' + repr(json.dumps(config)) + ''')
''' + inspect.getsource(unpack_sources) + inspect.getsource(evaluation_dependencies) + '''
work = Path('/kaggle/working')
repo = Path('/tmp/paper-baseline-source')
scope = Path('/tmp/official-SCOPE')
def checkout(url, path, revision):
    subprocess.run(['git','clone',url,str(path)],check=True)
    subprocess.run(['git','checkout','--detach',revision],cwd=path,check=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
    if actual != revision:
        raise ValueError('Upstream source revision mismatch')
checkout('https://github.com/munnn01/t_scope_v3.git',repo,CONFIG['base_revision'])
checkout('https://github.com/miccunifi/SCOPE.git',scope,CONFIG['scope_revision'])
payload=json.loads(zlib.decompress(base64.b64decode(''' + repr(encoded) + ''')))
unpack_sources(payload,repo)
subprocess.run([sys.executable,'-m','pip','install','--quiet','uv==0.9.9'],check=True)
subprocess.run([sys.executable,'-m','uv','python','install','3.11.14'],check=True)
runtime='/tmp/paper-baseline-env'
subprocess.run([sys.executable,'-m','uv','venv','--python','3.11.14',runtime],check=True)
python=runtime+'/bin/python'
subprocess.run([sys.executable,'-m','uv','pip','install','--python',python,'-r',str(repo/'requirements.txt')],check=True)
if CONFIG['mode']=='eval':
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',python,*evaluation_dependencies(python,str(repo))],check=True)
env=os.environ.copy()
env['PYTHONPATH']=str(repo)
env['PYTHONUNBUFFERED']='1'
env['PYTHONDONTWRITEBYTECODE']='1'
env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
find="from scripts.prepare_kaggle_datasets import find_mounted_kaggle_dataset; print(find_mounted_kaggle_dataset())"
data=subprocess.check_output([python,'-c',find],cwd=repo,env=env,text=True).strip().splitlines()[-1]
if data=='None' or not Path(data).is_dir():
    raise RuntimeError('Actual mounted MOT17 dataset is required')
command=[python,str(repo/'scripts/run_paper_baselines.py'),'--mode',CONFIG['mode'],
         '--data-root',data,'--output-dir',str(work),'--scope-root',str(scope)]
if CONFIG.get('job'):
    job=CONFIG['job']
    job_file=work/(job['job_id']+'.json')
    job_file.write_text(json.dumps(job,indent=2),encoding='utf-8')
    command+=['--job-file',str(job_file)]
    if CONFIG['mode'] in ('canary','train'):
        priors=list(Path('/kaggle/input').rglob('shared_entropy.pt'))
        if len(priors)!=1 or hashlib.sha256(priors[0].read_bytes()).hexdigest()!=job['entropy_sha256']:
            raise ValueError('Missing, ambiguous or mismatched shared entropy input')
        command+=['--entropy',str(priors[0])]
    for option,suffix in (('resume','.last.pt'),('checkpoint','.last.pt')):
        if CONFIG.get(option):
            matches=list(Path('/kaggle/input').rglob(job['job_id']+suffix))
            if len(matches)!=1 or hashlib.sha256(matches[0].read_bytes()).hexdigest()!=CONFIG[option]['sha256']:
                raise ValueError('Checkpoint input hash mismatch')
            command+=['--'+option,str(matches[0])]
print('PAPER_INPUT_VERIFIED '+json.dumps({'mode':CONFIG['mode'],'sources':payload['hashes']}),flush=True)
subprocess.run(command,cwd=repo,env=env,check=True)
'''


def read_logs(api, kernel_id, live=False):
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    owner, slug = kernel_id.split("/")
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        result = client.kernels.kernels_api_client.list_kernel_session_output(request)
        content = result.log or ""
        if not content and live:
            http = client._http_client
            http._init_session()
            headers = dict(http._session.headers)
            headers["Accept"] = "text/event-stream, */*"
            headers.pop("Content-Type", None)
            parts = []
            started = time.monotonic()
            try:
                with http._session.get(http._endpoint+"/v1/kernels/logs/stream/"+kernel_id,
                    headers=headers, auth=http._session.auth, stream=True, timeout=(10, 6)) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.startswith("data:"):
                            value = line[5:].strip()
                            if value == "END_OF_LOG":
                                break
                            try:
                                parts.append(str(json.loads(value).get("data", "")))
                            except ValueError:
                                parts.append(value)
                        if time.monotonic()-started > 10:
                            break
            except Exception:
                pass
            content = "".join(parts)
    try:
        values = json.loads(content)
        if isinstance(values, list):
            return "".join(str(v.get("data", "")) for v in values)
    except ValueError:
        pass
    return content


def admission_check(manifest, row):
    if row["config"]["mode"] != "train":
        return
    from scripts.run_paper_baselines import METHODS, protocol_hash
    passed = [r for r in manifest["jobs"] if r.get("canary_verified") is True and r["status"] == "complete"]
    if {r["config"].get("job", {}).get("method") for r in passed} != set(METHODS):
        raise ValueError("Production requires both verified real-data canaries")
    expected = protocol_hash(row["config"]["job"])
    if any(r.get("verified_protocol_sha256") != expected for r in passed):
        raise ValueError("Production sources/conditions differ from verified canaries")


def active_workloads(api, username):
    """Read all owned kernel pages; do not admit an account already at capacity."""
    token, seen, active = None, set(), []
    from scripts.kaggle_v3 import _http_status_codes
    while True:
        response = api.kernels_list_with_response(mine=True, user=username, page_size=100, page_token=token)
        for kernel in response.kernels:
            ref = getattr(kernel, "ref", "")
            if not ref:
                continue
            if ref.split("/")[0].casefold() != username.casefold():
                raise ValueError("Unexpected account in owned-kernel listing")
            try:
                status = str(api.kernels_status(ref).status).split(".")[-1].lower()
            except Exception as error:
                # Deleted notebooks can remain in the owner listing. A404
                # establishes absence; authorization/transport errors still fail.
                if _http_status_codes(error) == {404}:
                    continue
                raise
            if status in ("running", "queued", "pending"):
                active.append(ref)
        token = response.next_page_token
        if not token:
            return active
        if token in seen:
            raise ValueError("Repeated workload pagination token")
        seen.add(token)


def verify_canary(row):
    import torch
    from scripts.run_paper_baselines import finite_tensors, protocol_hash
    directory = Path(row["output_directory"])
    name = row["config"]["job"]["job_id"]
    result = json.loads((directory / (name+".result.json")).read_text(encoding="utf-8"))
    checkpoint = directory / (name+".canary.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    job = json.loads((directory / (name+".json")).read_text(encoding="utf-8"))
    if (state["job"] != job or state["status"] != "canary_passed" or state["optimizer_updates"] < 128
            or result["status"] != "canary_passed" or not finite_tensors(state)
            or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != result["checkpoint_sha256"]
            or result["gpu"] != "Tesla T4" or result["protocol_sha256"] != protocol_hash(job)):
        raise ValueError("Canary checkpoint evidence failed verification")
    row.update(canary_verified=True, verified_protocol_sha256=protocol_hash(job))


def operate(path, action, selected=(), pool_file=None):
    from scripts.kaggle_v3 import (load_account_pool, get_authenticated_api, _quiet_api,
        _remaining_hours, _http_status_codes, _owned_kernel_absent)
    from scripts.kaggle_output_download import download_kernel_outputs
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if action == "push" and manifest.get("export_approved") is not True:
        raise ValueError("Explicit generated-code export approval is required before Kaggle submission")
    if pool_file is None:
        raise ValueError("Supply a private pool file through --pool-file")
    accounts = dict(load_account_pool(pool_file))
    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(path)
    for row in manifest["jobs"]:
        if selected and row["id"] not in selected:
            continue
        stage = "authenticate"
        try:
            with _quiet_api():
                api = get_authenticated_api(row["username"], accounts[row["username"]])
            if action == "push":
                if row["status"] != "planned":
                    continue
                admission_check(manifest, row)
                with _quiet_api():
                    stage = "quota"
                    remaining = _remaining_hours(api)
                    if remaining < row.get("minimum_gpu_hours", 1):
                        raise ValueError("Insufficient current GPU quota")
                    stage = "existing_workloads"
                    active = active_workloads(api, row["username"])
                    row["other_active_workloads"] = active
                    if len(active) >= 3:
                        raise ValueError("Account already has three active workloads")
                    stage = "target_absence"
                    try:
                        api.kernels_status(row["kernel_id"])
                    except Exception as error:
                        if 404 not in _http_status_codes(error):
                            if _http_status_codes(error) != {403} or not _owned_kernel_absent(api, row, accounts[row["username"]]):
                                raise ValueError("Kernel absence could not be verified")
                    else:
                        raise ValueError("Refusing to overwrite existing kernel")
                folder = path.parent / "workers" / row["id"]
                folder.mkdir(parents=True, exist_ok=True)
                code = build_worker(row["config"])
                (folder / "worker.py").write_text(code, encoding="utf-8")
                metadata = {"id": row["kernel_id"], "title": row["kernel_id"].split("/")[1],
                    "code_file": "worker.py", "language": "python", "kernel_type": "script",
                    "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
                    "enable_internet": "true", "dataset_sources": ["ahmedsamir1598/mot17challenge"],
                    "kernel_sources": row.get("kernel_sources", []), "competition_sources": [], "model_sources": []}
                (folder / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                row.update(status="submitting", worker_sha256=hashlib.sha256(code.encode()).hexdigest())
                save()
                stage = "private_kernel_submission"
                with _quiet_api():
                    result = api.kernels_push(str(folder), acc="NvidiaTeslaT4")
                if getattr(result, "error", ""):
                    raise RuntimeError("Kaggle rejected baseline worker")
                row.update(status="submitted", quota_at_push=remaining,
                    submitted_at_utc=datetime.now(timezone.utc).isoformat())
            else:
                stage = "status_and_logs"
                with _quiet_api():
                    result = api.kernels_status(row["kernel_id"])
                    row["status"] = str(result.status).split(".")[-1].lower()
                    logs = read_logs(api, row["kernel_id"], row["status"] == "running")
                for token in accounts.values():
                    logs = logs.replace(token, "[REDACTED]")
                logdir = path.parent / "logs"
                logdir.mkdir(exist_ok=True)
                (logdir / (row["id"] + ".log")).write_text(logs, encoding="utf-8")
                markers = [line for line in logs.splitlines() if "PAPER_" in line]
                row["last_progress"] = markers[-1] if markers else None
                row["log_tail"] = logs.splitlines()[-10:]
                if action == "pull" and row["status"] in ("complete", "error") and not row.get("output_directory"):
                    destination = path.parent / "results" / row["id"]
                    destination.mkdir(parents=True, exist_ok=False)
                    with _quiet_api():
                        download_kernel_outputs(api, row["kernel_id"], destination)
                    row["output_directory"] = str(destination.resolve())
                    if row["status"] == "complete" and row["config"]["mode"] == "canary":
                        verify_canary(row)
            for key in ("error_type", "error_stage", "http_status"):
                row.pop(key, None)
        except Exception as error:
            row["error_type"] = type(error).__name__
            row["error_stage"] = stage
            row["http_status"] = sorted(_http_status_codes(error))
        row["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        summary = {k: row.get(k) for k in ("id", "kernel_id", "status", "last_progress", "error_type", "error_stage", "http_status", "output_directory")}
        if row["status"] == "error":
            summary["log_tail"] = row.get("log_tail")
        print(json.dumps(summary), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("push", "status", "pull"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pool-file", required=True, help="Private account-pool JSON; never copied into workers")
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    operate(args.manifest, args.action, args.only, pool_file=args.pool_file)
