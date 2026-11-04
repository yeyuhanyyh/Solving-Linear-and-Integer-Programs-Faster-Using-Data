"""Explicit ND eligibility-gate adapter around unchanged frozen v4 functions.

No monkeypatching, model loading or computation occurs at import. Both widths
run the original training-only1200-update gate. Only passing widths enter the
unchanged external-validation core; two failures prevent a completed model.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import time
import bundle_common as common
import nd_controls as controls


def eligible_widths(report):
    records = report.get('records', [])
    if ([r.get('width') for r in records] != [32, 64] or
        not report.get('control_complete') or
        any(r.get('diagnostic', {}).get('steps') != 1200 for r in records)):
        raise controls.ControlIncomplete('Both candidate gates must execute all1200 updates')
    return tuple(r['width'] for r in records if r['diagnostic'].get('passed') is True)


def fit_nd_with_gate(api, case, validation, output, device):
    """Return the same NeuralLearner/save format as api.fit, with real gates."""
    import neural_support as data
    import neural_diving_baseline as nd
    from nd_training import fit_nd_complete, save_nd_complete
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError('Preserve every prior fit attempt; this adapter requires fresh output')
    data.validate_training_pair(case, validation)
    started = time.perf_counter()
    config = nd.NeuralDivingConfig(seed=int(case.metadata['baseline_seed']), device=device)
    controls.check_config(config, diagnostic=True)
    guard = dict(method='neural_diving', training_hash=data.training_hash(case),
        validation_hash=data.training_hash(validation), source_sha256=data.implementation_fingerprint(),
        device=device, configuration=asdict(config), explicit_gate_adapter_sha256=common.sha(__file__),
        gate_control_source_sha256=common.sha(controls.__file__), test_information_used=False)
    data.write_json(destination/'fit_guard.json', guard)
    diagnostic_folder = destination/'gate-diagnostic'
    diagnostic_folder.mkdir()
    gate_started, gate_cpu = time.perf_counter(), time.process_time()
    gate = controls.diagnostic(case, device, diagnostic_folder)
    gate.update(learning_wall_seconds=time.perf_counter()-gate_started,
        learning_cpu_seconds=time.process_time()-gate_cpu)
    data.write_json(destination/'gate-diagnostic.json', gate)
    widths = eligible_widths(gate)
    if not widths:
        # The full failed diagnostic is already durable; no warmup/refit occurs.
        raise nd.TrainingSanityError('Both candidate widths failed the original1200-step sanity gate', gate)
    selected_config = replace(config, width_candidates=widths)
    model = fit_nd_complete(case, validation, device=device, config=selected_config)
    raw = model.training_metadata
    core_seconds = float(raw['training_seconds'])
    candidate_gates = {r['width']: r['diagnostic'] for r in gate['records']}
    passing_candidates = {r['width']: r for r in raw['width_candidates']}
    all_candidates = []
    for width in (32, 64):
        candidate = dict(passing_candidates.get(width, dict(width=width, eligible=False,
            reason='Original training-only1200-step optimization/integration gate failed')))
        candidate['sanity_gate'] = candidate_gates[width]
        all_candidates.append(candidate)
    total = time.perf_counter()-started
    overhead = total-core_seconds-gate['learning_wall_seconds']
    if overhead < -1e-5:
        raise ValueError('Gate and core clocks overlap')
    raw = dict(raw, training_seconds=total, core_training_seconds=core_seconds,
        discarded_gate_seconds=gate['learning_wall_seconds'],
        adapter_preparation_and_bookkeeping_seconds=max(0., overhead),
        optional_overfit_qa_run=True, optional_overfit_qa_seconds=gate['learning_wall_seconds'],
        width_candidates=all_candidates,
        sanity_gate=dict(passed=True, scope='training_only', eligible_widths=list(widths),
            width_candidates=[r['diagnostic'] for r in gate['records']]),
        gate_diagnostic=gate, gate_weights_discarded=True, gate_feature_cache_reused=False,
        explicit_gate_adapter_sha256=common.sha(__file__),
        original_candidate_widths=[32, 64], gate_passed_candidate_widths=list(widths),
        adaptation='Original1200-step training-only gate excludes failing widths. The unchanged v4 complete-predictor core selects among passing widths with external validation64, then refits all training rows. Gate and core preparation are included in learning time.')
    model.training_metadata = raw
    model.to('cpu')
    save_nd_complete(model, destination/'complete-predictor.pt')
    metadata = dict(method='neural_diving', fit_wall_seconds=time.perf_counter()-started,
        learning_time_seconds=total, learning_time_includes_diagnostic=True,
        discarded_diagnostic=gate, training_count=case.n_train,
        external_validation_count=validation.n_train, training_hash=data.training_hash(case),
        validation_hash=data.training_hash(validation), target_K=int(case.affine['rank']),
        learning_time_excludes_shared_label_generation=True,
        learning_time_includes='Both original1200-step eligibility gates and all feature preparation, external-validation selection, final fitting and readiness checks',
        fit_wall_time_additionally_includes='Deployment transfer and checkpoint serialization',
        original_algorithm=False, matched_dimension_rule='Exactly D-K common commitment coordinates are fixed',
        matched_inference_is_explicit_adaptation=True, raw_training_metadata=raw,
        explicit_gate_adapter_sha256=common.sha(__file__))
    data.write_json(destination/'metadata.json', metadata)
    artifacts = {p.name:data.sha(p) for p in destination.iterdir() if p.is_file()}
    data.write_json(destination/'neural-complete.json', dict(training_hash=metadata['training_hash'],
        method='neural_diving', source_sha256=guard['source_sha256'], artifact_sha256=artifacts,
        explicit_gate_adapter_sha256=common.sha(__file__)))
    return api.NeuralLearner(case, 'neural_diving', model, metadata)
