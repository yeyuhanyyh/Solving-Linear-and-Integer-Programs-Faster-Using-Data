"""Shared guards and artifact helpers for the portable v2 runners."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import hashlib, json, platform, sys
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
import adapter

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE / "controls"))
METHODS = ("full", "ours", "svm_native", "mlopt_native", "neural_diving_native",
           "neural_diving_matched", "li_bounds_residual")
MODEL_KEYS = ("ours", "svm_native", "mlopt_native", "neural_diving", "li_bounds_residual")
MODEL_FOR_METHOD = {m: ("neural_diving" if m.startswith("neural_diving_") else m) for m in METHODS if m != "full"}
VERSION = "independent-confirmation-v2-20261009"


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".writing")
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def event(**values: Any) -> None:
    print(json.dumps(clean(dict(utc=utc(), **values)), ensure_ascii=True, allow_nan=False), flush=True)


def selected(values: list[str] | None, allowed: tuple[str, ...] | list[str]) -> list[str]:
    if values is None:
        return list(allowed)
    if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
        raise ValueError(f"Unsupported, empty, or duplicated selection: {values}")
    return list(values)


def seed_for(case_id: str) -> int:
    return int.from_bytes(hashlib.sha256((VERSION+":"+case_id).encode()).digest()[:4], "little") % (2**31-1)


def runtime() -> dict[str, Any]:
    import scipy, sklearn, highspy, torch
    from threadpoolctl import threadpool_info
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)
    gpu = torch.cuda.is_available()
    cpu_model=platform.processor()
    cpuinfo=Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model=line.split(":",1)[1].strip()
                break
    return dict(python=sys.version, platform=platform.platform(), hostname=platform.node(),machine=platform.machine(),
        cpu_model=cpu_model,hostname_role="one host shared by all timed methods",
        cpu_count=os.cpu_count(), versions=dict(numpy=np.__version__, scipy=scipy.__version__,
        sklearn=sklearn.__version__, highs=highspy.Highs().version(), torch=torch.__version__),
        cuda_available=gpu, cuda_runtime=torch.version.cuda,
        cuda_device=(torch.cuda.get_device_name(0) if gpu else None), solver_threads=1,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        timed_prediction_device="cpu", tensor_threads=torch.get_num_threads(),
        tensor_interop_threads=torch.get_num_interop_threads(),blas_threads=1,
        threadpools=threadpool_info())


def resolved_configs(case_id: str, protocol: dict[str, Any], device: str) -> dict[str, Any]:
    from mlopt_baseline import Config
    from neural_diving_baseline import NeuralDivingConfig
    from preference_baseline import PreferenceConfig
    common = protocol.get("baseline_configs", {})
    seed = seed_for(case_id)
    def make(cls, key):
        values = dict(common.get(key, {}))
        values.update(seed=seed, device=device)
        return asdict(cls(**values))
    return clean(dict(ours=dict(learner="unchanged exact fit_affine", training_count=120),
        svm_native=dict(learner="unchanged XQAAffineBaseline", random_state=int(case_id.split("_")[1][4:]),
                        C=1., kernel="linear", training_count=120),
        mlopt_native=make(Config,"mlopt_native"), neural_diving=make(NeuralDivingConfig,"neural_diving"),
        li_bounds_residual=make(PreferenceConfig,"li_bounds_residual")))


def source_hashes() -> dict[str, str]:
    # This includes dimensions, solver/physics, fitting and online timing code.
    # nd_root_lp_features.py is inference code and is included by this scan.
    # The local exporter and reporting script cannot affect fitted predictions.
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(CODE.rglob("*.py"))
            if not p.name.startswith("test_") and p.name not in ("export_dataset.py", "build_runtime_bundle.py", "summarize_results.py", "verify_package.py")}


def freeze(device: str | None = None, create: bool = True) -> dict[str, Any]:
    path = ROOT / "models" / "training-freeze.json"
    protocol = read(ROOT / "protocol.json")
    manifest = adapter.dataset_manifest(verify=False)
    saved = read(path) if path.exists() else None
    actual_device = device or (saved["device"] if saved is not None else protocol.get("device", "cuda"))
    payload = dict(version=VERSION, source_sha256=source_hashes(), protocol_sha256=digest(ROOT/"protocol.json"),
        dataset_fingerprint=manifest["dataset_fingerprint"], device=actual_device,
        resolved_configs={cid: resolved_configs(cid, protocol, actual_device) for cid in adapter.list_case_ids()},
        methods=list(METHODS), confirmation_generation_policy="after all fitted model checkpoints are frozen")
    payload["fingerprint"] = object_digest(payload)
    if saved is not None and saved != payload:
        raise RuntimeError("Code, protocol, data, or device changed after training freeze; preserve this run and use a new package")
    if saved is None:
        if not create:
            raise RuntimeError("Training configuration has not been frozen")
        dump(path, payload)
    return payload


def artifact_hashes(directory: Path) -> dict[str, str]:
    return {p.relative_to(directory).as_posix(): digest(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name!="complete.json" and not p.name.endswith(".writing")}


def model_guard(case: adapter.FrozenCase, model_key: str, frozen: dict[str, Any]) -> dict[str, Any]:
    return dict(case_id=case.case_id, model_key=model_key, training_count=120,
        dataset_fingerprint=case.dataset_fingerprint, freeze_fingerprint=frozen["fingerprint"],
        config=frozen["resolved_configs"][case.case_id][model_key], test_data_seen_by_fit=False)


def checkpoint(case: adapter.FrozenCase, model_key: str, frozen: dict[str, Any], required: bool = True) -> dict[str, Any] | None:
    folder = ROOT / "models" / case.case_id / model_key
    marker = folder / "complete.json"
    if not marker.exists():
        if required:
            raise RuntimeError(f"Missing completed offline model: {case.case_id}/{model_key}")
        return None
    value = read(marker)
    if value["guard"] != model_guard(case, model_key, frozen) or value["artifact_sha256"] != artifact_hashes(folder):
        raise RuntimeError(f"Offline model was modified after completion: {case.case_id}/{model_key}")
    if model_key in ("neural_diving", "li_bounds_residual") and not value["sanity_gate_passed"]:
        raise RuntimeError(f"Unpassed training-only sanity gate: {case.case_id}/{model_key}")
    return value


class PhaseLock:
    def __init__(self, phase: str):
        self.phase, self.fd = phase, None
        self.path = ROOT / "logs" / "compute-phase.lock"
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd=os.open(self.path, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"Computation already active: {self.path.read_text()}; verify owner before removing a stale lock")
        os.write(self.fd,json.dumps(dict(phase=self.phase,pid=os.getpid(),utc=utc())).encode())
        os.fsync(self.fd)
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
            self.path.unlink()


def ensure_cpu(model: Any) -> None:
    # ND exposes .to(); Li already returns its trained network on CPU.
    if hasattr(model,"to"):
        model.to("cpu")
    import torch
    seen=set()
    def walk(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value,torch.nn.Module):
            value.to("cpu").eval()
            if any(p.device.type!="cpu" for p in value.parameters()):
                raise RuntimeError("A timed model retains GPU parameters")
        elif isinstance(value,torch.Tensor):
            if value.device.type!="cpu":
                raise RuntimeError("A timed model retains GPU tensors")
        elif isinstance(value,(list,tuple,dict)):
            for item in (value.values() if isinstance(value,dict) else value):
                walk(item)
        elif hasattr(value,"__dict__"):
            for item in vars(value).values():
                walk(item)
    walk(model)
