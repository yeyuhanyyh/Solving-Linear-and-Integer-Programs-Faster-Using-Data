"""Fitting clocks for the frozen grouped-knapsack ND/Preference learners.

This module does not train on import. It instruments function source in memory;
it never changes a frozen learner file, seed, grid, stopping rule or model state.
Use one scientific fitting job per process/device. Shared labels and input data
must already be loaded before calling the fitting helpers.
"""
from __future__ import annotations

import ast
import contextlib
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
import textwrap
import time
from typing import Any


def synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _call_name(node.value) + "." + node.attr
    return ast.dump(node, include_attributes=False)


SCIENTIFIC_ND_CALLS = {
    "nd.NeuralDivingConfig", "nd._setup", "nd.FixedBipartiteGraph",
    "graph.precompute_training_root_lp", "nd._prepare_pools", "nd._sanity_gate",
    "nd._warmup_selection", "nd._new_network", "predictor.load_state_dict",
    "predictor.eval", "nd.NeuralDivingModel", "nd.solve_matched_cost",
    "nd._quality", "nd._optimizer", "nd._train_epoch", "nd._full_diagnostics",
    "np.random.default_rng", "replace", "torch.cuda.empty_cache",
}


def scientific_nd_calls(source: str) -> list[tuple[str, str]]:
    """Exact call/argument inventory, excluding only timing/metadata operations."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in SCIENTIFIC_ND_CALLS:
                found.append((name, ast.dump(node, include_attributes=False)))
    return found


def build_instrumented_nd_source(original: str) -> tuple[str, dict[str, Any]]:
    source = textwrap.dedent(original)
    edits = [
        (
            "    candidates=[]; chosen=None; best_score=None; selection_start=time.perf_counter()\n",
            "    diagnostic_gate_seconds=0.0\n"
            "    validation_call_seconds=0.0\n"
            "    diagnostic_gate_clocks=[]\n"
            "    _fit_alignment_synchronize(device)\n"
            "    candidates=[]; chosen=None; best_score=None; selection_start=time.perf_counter()\n",
        ),
        (
            "        gate=nd._sanity_gate(graph,candidate_config,case,costs,pools,training_indices,reference,adapter.solve)\n",
            "        _fit_alignment_synchronize(device)\n"
            "        diagnostic_gate_start=time.perf_counter()\n"
            "        gate=nd._sanity_gate(graph,candidate_config,case,costs,pools,training_indices,reference,adapter.solve)\n"
            "        _fit_alignment_synchronize(device)\n"
            "        diagnostic_gate_elapsed=time.perf_counter()-diagnostic_gate_start\n"
            "        diagnostic_gate_seconds+=diagnostic_gate_elapsed\n"
            "        diagnostic_gate_clocks.append(dict(width=int(width),seconds=diagnostic_gate_elapsed,passed=bool(gate['passed'])))\n",
        ),
        (
            "        records=[nd.solve_matched_cost(shell,case,costs[int(i)],config.seed+300001+int(i)*4099,\n",
            "        _fit_alignment_synchronize(device)\n"
            "        validation_call_start=time.perf_counter()\n"
            "        records=[nd.solve_matched_cost(shell,case,costs[int(i)],config.seed+300001+int(i)*4099,\n",
        ),
        (
            "        quality=nd._quality(records,reference,validation_solver)\n",
            "        _fit_alignment_synchronize(device)\n"
            "        validation_call_seconds+=time.perf_counter()-validation_call_start\n"
            "        quality=nd._quality(records,reference,validation_solver)\n",
        ),
        (
            "    selection_seconds=time.perf_counter()-selection_start\n",
            "    _fit_alignment_synchronize(device)\n"
            "    selection_seconds=time.perf_counter()-selection_start\n",
        ),
        (
            "        feature_preparation_seconds=feature_seconds,epoch_selection_seconds=selection_seconds,final_refit_seconds=refit_seconds,\n",
            "        feature_preparation_seconds=feature_seconds,epoch_selection_seconds=selection_seconds,final_refit_seconds=refit_seconds,\n"
            "        diagnostic_gate_seconds=diagnostic_gate_seconds,diagnostic_gate_clocks=diagnostic_gate_clocks,\n"
            "        selection_validation_call_seconds=validation_call_seconds,\n"
            "        epoch_selection_without_diagnostics_seconds=selection_seconds-diagnostic_gate_seconds,\n"
            "        fitting_only_seconds=selection_seconds-diagnostic_gate_seconds+refit_seconds,\n"
            "        fitting_without_validation_solves_seconds=selection_seconds-diagnostic_gate_seconds-validation_call_seconds+refit_seconds,\n",
        ),
    ]
    patched = source
    for old, new in edits:
        if patched.count(old) != 1:
            raise ValueError("Frozen ND timing anchor is missing or nonunique: " + old)
        patched = patched.replace(old, new)
    # Every pre-existing scientific call, including seed expressions, is exact.
    before_calls = scientific_nd_calls(source)
    after_calls = scientific_nd_calls(patched)
    if before_calls != after_calls:
        raise AssertionError("ND instrumentation changed a scientific call/argument")
    restored = patched
    for old, new in reversed(edits):
        if restored.count(new) != 1:
            raise AssertionError("Timing transformation is not exactly reversible")
        restored = restored.replace(new, old)
    if restored != source:
        raise AssertionError("Unexpected edit outside timing additions")
    compile(patched, "<grouped-ND-fitting-clocks>", "exec")
    return patched, dict(
        original_function_sha256=_sha_text(source),
        instrumented_function_sha256=_sha_text(patched),
        exact_reversal_passed=True,
        scientific_call_arguments_unchanged=True,
        scientific_calls_checked=len(before_calls),
        change_scope="CUDA synchronization, independent clocks and metadata only",
    )


@dataclass
class FitResult:
    model: Any
    fitting_seconds: float
    clocks: dict[str, Any]
    provenance: dict[str, Any]


def fit_neural_diving(shared_trainer: Any, case: Any, device: str = "cpu") -> FitResult:
    """Original ND fit with exact diagnostic/root-feature exclusion.

    The primary clock equals architecture/epoch selection excluding discarded
    1200-step diagnostic gates, plus final refit. Selection's validation calls
    remain included, with an additional exact clock available if the common
    protocol elects to exclude them. Final export is a separate helper below.
    """
    original = inspect.getsource(shared_trainer.fit_nd_complete)
    patched, provenance = build_instrumented_nd_source(original)
    namespace = dict(shared_trainer.__dict__)
    namespace["_fit_alignment_synchronize"] = synchronize
    exec(compile(patched, "<grouped-ND-fitting-clocks>", "exec"), namespace)
    fitter = namespace["fit_nd_complete"]
    synchronize(device)
    start = time.perf_counter()
    model = fitter(case, device=device)
    synchronize(device)
    fit_call_seconds = time.perf_counter() - start
    metadata = model.training_metadata
    if not metadata.get("sanity_gate", {}).get("passed"):
        raise ValueError("The unchanged ND readiness gate failed")
    clocks = {name: float(metadata[name]) for name in (
        "feature_preparation_seconds", "epoch_selection_seconds", "final_refit_seconds",
        "diagnostic_gate_seconds", "selection_validation_call_seconds",
        "epoch_selection_without_diagnostics_seconds", "fitting_only_seconds",
        "fitting_without_validation_solves_seconds",
    )}
    if clocks["fitting_only_seconds"] < 0 or clocks["fitting_without_validation_solves_seconds"] < 0:
        raise AssertionError("Nonpositive phase accounting")
    clocks["fit_call_wall_seconds"] = fit_call_seconds
    clocks["export_included"] = False
    clocks["root_lp_graph_preparation_included"] = False
    clocks["discarded_diagnostic_gates_included"] = False
    return FitResult(model, clocks["fitting_only_seconds"], clocks, provenance)


def export_neural_diving(shared_trainer: Any, model: Any, destination: str | Path,
                         device: str = "cpu") -> float:
    synchronize(device)
    start = time.perf_counter()
    shared_trainer.save_nd_complete(model, Path(destination))
    synchronize(device)
    return time.perf_counter() - start


def _extract_preference_fit_prefix(shared_trainer: Any) -> tuple[Any, dict[str, Any]]:
    """Reuse the original Li branch through fit/adapter metadata, before export."""
    source = textwrap.dedent(inspect.getsource(shared_trainer.train_one))
    tree = ast.parse(source)
    branch = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "method"
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "li"):
            branch = node.body
            break
    if branch is None:
        raise ValueError("Cannot find frozen Preference Learning fit branch")
    prefix = []
    for statement in branch:
        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                and _call_name(statement.value.func) == "model.save"):
            break
        prefix.append(statement)
    else:
        raise ValueError("Cannot locate final Preference checkpoint export")
    if not any(isinstance(n, ast.ClassDef) and n.name == "ExpandedCandidatePreference" for n in prefix):
        raise ValueError("Original top-k clipping adapter is missing")
    function = ast.FunctionDef(
        name="_fit_original_grouped_preference",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in ("case", "partial", "device")],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=prefix + [ast.Return(value=ast.Tuple(elts=[ast.Name(id="model", ctx=ast.Load()),
                                                        ast.Name(id="metadata", ctx=ast.Load())], ctx=ast.Load()))],
        decorator_list=[],
    )
    generated = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(shared_trainer.__dict__)
    exec(compile(generated, "<grouped-Preference-original-fit-branch>", "exec"), namespace)
    return namespace[function.name], dict(
        original_train_one_sha256=_sha_text(source),
        original_preference_branch_prefix_ast_sha256=_sha_text(ast.dump(ast.Module(body=prefix, type_ignores=[]), include_attributes=False)),
        scientific_branch_copied_verbatim_ast=True,
        branch_ends_before_original_final_save=True,
    )


@contextlib.contextmanager
def _time_automatic_saves(base_class: Any, device: str):
    original_save = base_class.save
    records: list[float] = []

    def timed_save(self, *args, **kwargs):
        synchronize(device)
        start = time.perf_counter()
        result = original_save(self, *args, **kwargs)
        synchronize(device)
        records.append(time.perf_counter() - start)
        return result

    base_class.save = timed_save
    try:
        yield records
    finally:
        base_class.save = original_save


def fit_preference_learning(shared_trainer: Any, case: Any, output_dir: str | Path,
                            device: str = "cpu") -> FitResult:
    """Use original class/config/fit. Its internal pre-save clock is primary.

    PreferenceBaseline.fit performs an automatic save after it freezes its
    internal fit_seconds. We retain that operation, time it independently, and
    do not count it in fitting time. A caller can perform the original trainer's
    additional final export with export_preference_learning below.
    """
    destination = Path(output_dir)
    if destination.exists() and any(p.name != "start.json" for p in destination.iterdir()):
        raise FileExistsError("Use a new Preference fitting directory (only start.json may already exist)")
    destination.mkdir(parents=True, exist_ok=True)
    fitter, provenance = _extract_preference_fit_prefix(shared_trainer)
    li = importlib.import_module("preference_baseline")
    synchronize(device)
    start = time.perf_counter()
    with _time_automatic_saves(li.PreferenceBaseline, device) as automatic_exports:
        model, metadata = fitter(case, destination, device)
    synchronize(device)
    call_wall = time.perf_counter() - start
    if not metadata.get("sanity_gate", {}).get("passed"):
        raise ValueError("The unchanged Preference Learning readiness gate failed")
    fit_seconds = float(metadata["fit_seconds"])
    clocks = dict(
        fitting_only_seconds=fit_seconds,
        fit_call_wall_seconds=call_wall,
        automatic_export_seconds=sum(automatic_exports),
        automatic_export_count=len(automatic_exports),
        export_included=False,
        shared_training_optimum_generation_included=False,
        method_specific_fitting_preparation_included=True,
    )
    for field in ("development_selection_seconds", "final_bank_preparation_seconds", "final_optimizer_search_seconds"):
        if field in metadata:
            clocks[field] = float(metadata[field])
    return FitResult(model, fit_seconds, clocks, provenance)


def export_preference_learning(model: Any, destination: str | Path,
                               device: str = "cpu") -> float:
    synchronize(device)
    start = time.perf_counter()
    model.save(Path(destination))
    synchronize(device)
    return time.perf_counter() - start


def _shared_trainer() -> Any:
    """Load the copied grouped source under a private, collision-free name.

    The runner must call its portable common.enable() before this helper so
    the unchanged source's adapter/runtime imports resolve to the bundle.
    """
    path = Path(__file__).resolve().parent / "source/generic/shared/train_uc_baselines.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    name = "_grouped_fit_alignment_shared_trainer"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError("Grouped shared-trainer module name collision")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load copied grouped shared trainer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fit_nd_complete(case: Any, device: str = "cpu", shared_trainer: Any = None) -> Any:
    """Runner convenience API, returning the original ND model type."""
    trainer = _shared_trainer() if shared_trainer is None else shared_trainer
    result = fit_neural_diving(trainer, case, device)
    result.model.training_metadata["fit_alignment_clock"] = result.clocks
    result.model.training_metadata["fit_alignment_provenance"] = result.provenance
    result.model.training_metadata["fit_alignment"] = dict(
        headline_fit_seconds=result.fitting_seconds,
        definition="epoch_selection_seconds - diagnostic_gate_seconds + final_refit_seconds",
        validation_call_time_retained=True,
        clocks=result.clocks,
        provenance=result.provenance,
        final_export_included=False,
    )
    return result.model


def fit_preference(case: Any, output_dir: str | Path, device: str = "cpu",
                   shared_trainer: Any = None) -> Any:
    """Runner convenience API, returning the original Preference model type."""
    trainer = _shared_trainer() if shared_trainer is None else shared_trainer
    result = fit_preference_learning(trainer, case, output_dir, device)
    result.model.metadata["fit_alignment_clock"] = result.clocks
    result.model.metadata["fit_alignment_provenance"] = result.provenance
    result.model.metadata["fit_alignment"] = dict(
        headline_fit_seconds=result.fitting_seconds,
        definition="Original PreferenceBaseline internal fit_seconds, frozen before automatic final save",
        clocks=result.clocks,
        provenance=result.provenance,
        final_export_included=False,
    )
    return result.model


def _state_digest(network: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def continue_nd(model: Any, case: Any, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    """Independent optional 2000-update control, never selected by test results.

    The supplied base model remains unchanged. The new model starts from its
    exact weights and uses the original control's SHA-derived batch seed,
    epoch-permutation batches, fresh Adam, clip5, and fixed final update2000.
    Additional fitting excludes root-feature preparation, periodic training-only
    diagnostics and export; their clocks are recorded separately.
    """
    import copy
    from dataclasses import asdict, replace
    import numpy as np
    import torch
    import neural_diving_baseline as nd
    import uc_adapters as data

    original_digest = _state_digest(model.full_predictor)
    original_shapes = {name: tuple(value.shape) for name, value in model.full_predictor.state_dict().items()}
    config = replace(model.config, device=device)
    if config.batch_size != 16 or config.gradient_clip != 5.0 or config.learning_rate != 0.001:
        raise ValueError("Original grouped ND optimizer convention changed")
    synchronize(device)
    preparation_start = time.perf_counter()
    costs = np.asarray(case.train_costs, dtype=float)
    graph = nd.FixedBipartiteGraph(case.model, costs, config.normalize_constraint_rows,
                                  device, config.use_root_lp_features,
                                  config.root_lp_time_limit, config.seed)
    graph.training_lp_cache = dict(model.graph.training_lp_cache)
    costs, pools, pool_metadata = nd._prepare_pools(case, data.singleton_pools(case), graph.integer_indices)
    root_features = graph.precompute_training_root_lp(costs)
    predictor = nd._new_network(graph, config, config.seed)
    predictor.load_state_dict(nd._state(model.full_predictor))
    continued = nd.NeuralDivingModel(graph, [], predictor, config, copy.deepcopy(model.training_metadata))
    nd._setup(config)
    synchronize(device)
    preparation_seconds = time.perf_counter() - preparation_start
    indices = np.arange(case.n_train, dtype=np.int64)
    batch_seed = int.from_bytes(hashlib.sha256((case.case_id + "|fixed-extra-optimization-budget-v1").encode()).digest()[:4], "little")
    rng = np.random.default_rng(batch_seed)
    order = np.empty(0, dtype=int)
    cursor = 0
    epochs_started = 0
    trajectory = []
    updates = []
    diagnostics_seconds = 0.0

    def record_diagnostics(step: int, learning_rate: float) -> None:
        nonlocal diagnostics_seconds
        synchronize(device)
        start = time.perf_counter()
        diagnostics = nd._full_diagnostics(predictor, costs, pools, indices)
        synchronize(device)
        diagnostics_seconds += time.perf_counter() - start
        trajectory.append(dict(step=step, learning_rate=learning_rate, diagnostics=diagnostics))

    synchronize(device)
    fit_start = time.perf_counter()
    optimizer = nd._optimizer(predictor, config)
    if optimizer.state:
        raise AssertionError("Continuation must begin with empty Adam state")
    record_diagnostics(0, 0.001)
    for step in range(1, 2001):
        if cursor >= len(order):
            order = rng.permutation(indices)
            cursor = 0
            epochs_started += 1
        chosen = order[cursor:cursor + 16]
        cursor += len(chosen)
        rate = 0.001 if step <= 1000 else 0.0001
        for group in optimizer.param_groups:
            group["lr"] = rate
        predictor.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = predictor(costs[chosen])
        loss = nd._full_loss(logits, chosen, pools)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite original full-predictor NLL")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)
        if not bool(torch.isfinite(gradient)):
            raise FloatingPointError("Nonfinite continuation gradient")
        optimizer.step()
        updates.append(dict(step=step, learning_rate=rate, batch_indices=chosen.tolist(),
                            batch_count=len(chosen), batch_NLL_before_update=float(loss.detach()),
                            unclipped_gradient_norm=float(gradient)))
        if step % 100 == 0:
            record_diagnostics(step, rate)
    predictor.eval()
    synchronize(device)
    fit_total_seconds = time.perf_counter() - fit_start
    if _state_digest(model.full_predictor) != original_digest:
        raise AssertionError("Original selected ND model was modified")
    if original_shapes != {name: tuple(value.shape) for name, value in predictor.state_dict().items()}:
        raise AssertionError("Continuation changed the selected architecture")
    metadata = dict(
        control="fixed extra optimization budget", config=asdict(config),
        optimizer_state_reset=True, initial_Adam_state_empty=True,
        optimizer_steps=2000, final_checkpoint_step=2000,
        all_training_count=case.n_train, train_cost_digest=nd._array_digest(costs),
        batch_seed=batch_seed, batching="original per-epoch permutation, batches16/16/4 for n36",
        learning_rate_schedule=[dict(steps=[1, 1000], learning_rate=0.001),
                                dict(steps=[1001, 2000], learning_rate=0.0001)],
        original_selected_width=config.hidden_size,
        initial_state_digest=original_digest, final_state_digest=_state_digest(predictor),
        preparation_seconds=preparation_seconds, fitting_with_diagnostics_seconds=fit_total_seconds,
        diagnostic_seconds=diagnostics_seconds,
        additional_fitting_seconds=fit_total_seconds - diagnostics_seconds,
        epochs_started=epochs_started, final_batch_cursor=cursor,
        pool_metadata=pool_metadata, root_features=root_features,
        trajectory=trajectory, updates=updates, training_only=True,
        validation_or_test_read=False, checkpoint_selected_by="fixed final update2000 only",
        original_selected_model_unchanged=True, export_included=False,
    )
    continued.training_metadata["fixed_extra_optimization_budget"] = metadata
    return continued, metadata


def static_audit(source_path: str | Path) -> dict[str, Any]:
    """No imports of scientific libraries, data loading, fitting or optimization."""
    path = Path(source_path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fit_nd_complete")
    original = ast.get_source_segment(source, function)
    assert original is not None
    _, result = build_instrumented_nd_source(original + "\n")
    result.update(shared_trainer_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  training_performed=False, optimization_performed=False)
    return result
