"""Prespecified ND controls, separate from unchanged v4 fitting.

Import is inert. Commands use training rows only and write fresh directories.
The context provider owns representation/certificate verification: no dimension
or training fingerprint is hardcoded here. A canonical bundle may copy this
file unchanged alongside its own verified bundle_common and fit_bundle.
"""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import asdict, replace
import hashlib
import math
import os
from pathlib import Path
import signal
import time
import traceback
import bundle_common as common
import fit_bundle as fitting

WIDTHS = (32, 64)
DIAGNOSTIC_STEPS = 1200
CONTINUATION_STEPS = 2000


class ControlIncomplete(RuntimeError):
    pass


def continuation_lr(step):
    if not isinstance(step, int) or isinstance(step, bool) or not 1 <= step <= 2000:
        raise ValueError('Continuation update must lie in 1..2000')
    return .001 if step <= 1000 else .0001


def continuation_seed(case_id):
    return int.from_bytes(hashlib.sha256((case_id + '|UC-fixed-extra2000').encode()).digest()[:4], 'little')


def check_config(config, *, diagnostic=False):
    expected = dict(graph_layers=6, batch_size=16, learning_rate=.001,
        weight_decay=0., gradient_clip=5., normalize_constraint_rows=True,
        use_root_lp_features=True, warmup_epochs=150, patience=20)
    if diagnostic:
        expected.update(sanity_train_count=8, sanity_steps=1200,
            sanity_decay_after_steps=400, sanity_learning_rate_decay=.1,
            sanity_solver_count=4, sanity_time_limit=5., sanity_fixed_fraction=.1,
            log_every=25)
    if config.hidden_size not in WIDTHS:
        raise ValueError('Only the frozen width32/64 architecture is permitted')
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError('Prespecified ND control changed: ' + name)


def accounting(core, diagnostic, additional=None):
    if not diagnostic.get('control_complete'):
        raise ControlIncomplete('The full two-width diagnostic was not completed')
    included = bool(core.get('original_metadata', {}).get('learning_time_includes_diagnostic', False))
    diagnostic_seconds = float(diagnostic['learning_wall_seconds'])
    values = dict(core_learning_wall_seconds=float(core['learning_wall_seconds']) - (diagnostic_seconds if included else 0.),
        discarded_diagnostic_wall_seconds=diagnostic_seconds)
    if any(not math.isfinite(x) or x < 0 for x in values.values()):
        raise ValueError('Invalid learning clock')
    values['base_learning_wall_seconds'] = sum(values.values())
    values['base_definition'] = 'Core fitting plus both discarded training-only eligibility gates, counted once; no diagnostic weights or feature cache reused.'
    values['diagnostic_all_passed'] = bool(diagnostic['all_diagnostics_passed'])
    if additional is not None:
        if not math.isfinite(additional) or additional < 0:
            raise ValueError('Invalid continuation clock')
        values['continuation_additional_wall_seconds'] = additional
        values['continuation_cumulative_wall_seconds'] = values['base_learning_wall_seconds'] + additional
    return values


@contextlib.contextmanager
def deadline(seconds):
    """Bound native unchanged diagnostics too; an interruption writes no seal."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive finite control budget is required')
    if os.name != 'posix':
        raise RuntimeError('Executing controls requires the Linux alarm guard')
    def expired(signum, frame):
        raise ControlIncomplete('The prespecified control exceeded its execution budget')
    old = signal.signal(signal.SIGALRM, expired)
    previous = signal.setitimer(signal.ITIMER_REAL, seconds)
    if previous[0] or previous[1]:
        signal.setitimer(signal.ITIMER_REAL, *previous)
        signal.signal(signal.SIGALRM, old)
        raise RuntimeError('Refuse to replace an existing process deadline')
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def validate_artifacts(folder, marker):
    folder = Path(folder).resolve()
    for relative, digest in marker['artifact_sha256'].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or path.is_symlink() or common.sha(path) != digest:
            raise ValueError('Bound artifact changed: ' + relative)


def validate_core(case, ctx, folder):
    folder = Path(folder).resolve()
    marker = common.load(folder / 'complete.json')
    if (marker.get('schema') != 'absolute-baseline-fit-complete-v1' or
        marker.get('method') != 'neural_diving' or
        marker.get('training_sha256') != case.dataset_fingerprint or
        marker.get('target_K') != int(case.affine['rank']) or
        marker.get('parent_bindings') != common.evidence_bindings(ctx) or
        marker.get('source_file_sha256') != common.sha(fitting.__file__) or
        marker.get('package_manifest_sha256') != common.verify_package() or
        marker.get('native_source_manifest_sha256') != common.V4_MANIFEST or
        marker.get('nd_original_sanity_gates_enforced') is not True or
        marker.get('nd_gate_adapter_sha256') != common.sha(Path(__file__).with_name('nd_gate_training.py')) or
        marker.get('nd_controls_sha256') != common.sha(__file__) or
        'fit.json' not in marker.get('artifact_sha256', {})):
        raise ValueError('Core ND fit is incomplete or belongs to another frozen case/source')
    validate_artifacts(folder, marker)
    checkpoint = folder / 'model' / 'complete-predictor.pt'
    if 'model/complete-predictor.pt' not in marker['artifact_sha256']:
        raise ValueError('Core seal does not bind the selected final weights')
    report = common.load(folder / 'fit.json')
    validate_gated_core_report(report)
    return report, checkpoint, dict(
        core_complete_sha256=common.sha(folder / 'complete.json'),
        core_weights_sha256=common.sha(checkpoint))


def validate_gated_core_report(report):
    metadata = report.get('original_metadata', {})
    gate = metadata.get('discarded_diagnostic', {})
    records = gate.get('records', [])
    if (metadata.get('learning_time_includes_diagnostic') is not True or
        metadata.get('explicit_gate_adapter_sha256') != common.sha(Path(__file__).with_name('nd_gate_training.py')) or
        gate.get('control_complete') is not True or
        [r.get('width') for r in records] != [32, 64] or
        any(r.get('diagnostic', {}).get('steps') != 1200 for r in records)):
        raise ValueError('Continuation requires a core fitted with both actual eligibility gates')
    passed = [r['width'] for r in records if r['diagnostic'].get('passed') is True]
    raw = metadata.get('raw_training_metadata', {})
    if (not passed or raw.get('selected_width') not in passed or
        raw.get('gate_passed_candidate_widths') != passed or
        raw.get('sanity_gate', {}).get('eligible_widths') != passed):
        raise ValueError('The selected core width did not pass its training-only gate')
    accounting(report, gate)
    return gate


def validate_diagnostic(case, ctx, folder):
    folder = Path(folder).resolve()
    marker = common.load(folder / 'complete.json')
    if (marker.get('schema') != 'absolute-nd-control-complete-v1' or
        marker.get('mode') != 'diagnostic' or
        marker.get('training_sha256') != case.dataset_fingerprint or
        marker.get('parent_bindings') != common.evidence_bindings(ctx) or
        marker.get('source_file_sha256') != common.sha(__file__) or
        marker.get('package_manifest_sha256') != common.verify_package()):
        raise ValueError('Diagnostic seal belongs to different data/source or is incomplete')
    validate_artifacts(folder, marker)
    report = common.load(folder / 'control.json')
    if not report.get('control_complete') or report.get('usable_checkpoint_saved'):
        raise ControlIncomplete('Both discarded diagnostics must finish before continuation')
    if [r['width'] for r in report['records']] != list(WIDTHS):
        raise ValueError('Both frozen candidate widths are required')
    if any(r['diagnostic'].get('steps') != DIAGNOSTIC_STEPS for r in report['records']):
        raise ControlIncomplete('A width did not execute all 1200 updates')
    return report


def diagnostic(case, device, output):
    """Same operations as frozen audit_overfit.audit; no returned weight/cache."""
    import numpy as np
    import adapter
    import neural_support as support
    import neural_diving_baseline as nd
    records = []
    for width in WIDTHS:
        start, cpu = time.perf_counter(), time.process_time()
        config = nd.NeuralDivingConfig(seed=int(case.metadata['baseline_seed']),
            device=device, hidden_size=width, width_candidates=(width,))
        check_config(config, diagnostic=True)
        nd._setup(config)
        graph = nd.FixedBipartiteGraph(case.model, case.train_costs,
            config.normalize_constraint_rows, device, config.use_root_lp_features,
            config.root_lp_time_limit, config.seed)
        graph.precompute_training_root_lp(case.train_costs[:config.sanity_train_count])
        costs, pools, _ = nd._prepare_pools(case, support.singleton_pools(case), graph.integer_indices)
        result = nd._sanity_gate(graph, config, case, costs, pools,
            np.arange(case.n_train), np.asarray(case.train_objectives), adapter.solve)
        fitting.synchronize(device)
        record = dict(width=width, config=asdict(config), diagnostic=result,
            learning_wall_seconds=time.perf_counter()-start,
            learning_cpu_seconds=time.process_time()-cpu,
            control_complete=result.get('steps') == DIAGNOSTIC_STEPS)
        common.write(output / ('width-%d.json' % width), record)
        records.append(record)
        del graph, pools, costs
        if not record['control_complete']:
            raise ControlIncomplete('Width%d returned before all 1200 diagnostic updates' % width)
    return dict(records=records, control_complete=True,
        all_diagnostics_passed=all(r['diagnostic'].get('passed') is True for r in records),
        usable_checkpoint_saved=False, discarded_weights_reused=False,
        discarded_feature_cache_reused=False,
        eligible_widths=[r['width'] for r in records if r['diagnostic'].get('passed') is True],
        interpretation='Discarded training-only eligibility gate. The canonical core adapter admits only widths with passed=True; two failures block formal fitting.')


def load_selected_weights(case, checkpoint, device):
    """Verify both identities before reconstructing the selected network on device."""
    import numpy as np
    import torch
    import neural_diving_baseline as nd
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('format_version') != 'uc-nd-complete-v1':
        raise ValueError('Wrong complete-predictor checkpoint format')
    metadata = payload['metadata']
    if not metadata['sanity_gate']['passed']:
        raise ValueError('Selected core predictor is not ready')
    if nd._array_digest(np.asarray(case.train_costs, dtype=float)) != metadata['train_cost_digest']:
        raise ValueError('Selected weights used a different training bank')
    config = replace(nd.NeuralDivingConfig(**payload['config']), device=device)
    check_config(config)
    if config.seed != int(case.metadata['baseline_seed']):
        raise ValueError('Baseline seed changed')
    nd._setup(config)
    graph = nd.FixedBipartiteGraph(case.model, case.train_costs,
        config.normalize_constraint_rows, device, config.use_root_lp_features,
        config.root_lp_time_limit, config.seed)
    if graph.matrix_digest != metadata['matrix_digest']:
        raise ValueError('Selected weights used a different physical formulation')
    predictor = nd._new_network(graph, config, config.seed)
    predictor.load_state_dict(payload['full_predictor_state'])
    predictor.eval()
    return nd.NeuralDivingModel(graph, [], predictor, config, metadata)


def update_loop(model, costs, pools, case, output):
    """Original fixed update rule, generalized only from 32 to current N labels."""
    import numpy as np
    import torch
    import neural_diving_baseline as nd
    config = model.config
    optimizer = nd._optimizer(model.full_predictor, config)
    if optimizer.state:
        raise ValueError('Continuation requires a newly reset Adam state')
    indices = np.arange(case.n_train)
    seed = continuation_seed(case.case_id)
    rng = np.random.default_rng(seed)
    order, cursor = np.empty(0, dtype=int), 0
    updates = []
    trajectory = [dict(step=0, diagnostics=nd._full_diagnostics(model.full_predictor, costs, pools, indices))]
    for step in range(1, CONTINUATION_STEPS + 1):
        if cursor >= len(order):
            order, cursor = rng.permutation(indices), 0
        chosen = order[cursor:cursor+config.batch_size]
        cursor += len(chosen)
        lr = continuation_lr(step)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.full_predictor.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model.full_predictor(costs[chosen])
        loss = nd._full_loss(logits, chosen, pools)
        if not bool(torch.isfinite(loss)):
            raise ControlIncomplete('Nonfinite continuation training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.full_predictor.parameters(), config.gradient_clip)
        if not bool(torch.isfinite(norm)):
            raise ControlIncomplete('Nonfinite continuation gradient')
        optimizer.step()
        updates.append(dict(step=step, batch_indices=chosen.tolist(), learning_rate=lr,
            NLL=float(loss.detach()), gradient_norm_before_clip=float(norm)))
        if step % 100 == 0:
            row = dict(step=step, diagnostics=nd._full_diagnostics(model.full_predictor, costs, pools, indices))
            trajectory.append(row)
            # Append-only progress cannot become an evaluable model.
            common.write(output / ('progress-%04d.json' % step), dict(row,
                recent_updates=updates[-100:], checkpoint_saved=False))
    model.full_predictor.eval()
    return dict(optimizer_steps=len(updates), final_checkpoint_step=CONTINUATION_STEPS,
        batch_seed=seed, optimizer_state_reset=True, original_selected_width=config.hidden_size,
        all_training_count=case.n_train, batch_size=config.batch_size,
        learning_rate_schedule={'1..1000': .001, '1001..2000': .0001},
        selection='Last update2000 weights only; no intermediate or test selection',
        updates=updates, trajectory=trajectory)


def continuation(case, checkpoint, device, output):
    import neural_support as support
    import nd_training
    import neural_diving_baseline as nd
    model = load_selected_weights(case, checkpoint, device)
    costs, pools, pool_metadata = nd._prepare_pools(case,
        support.singleton_pools(case), model.graph.integer_indices)
    root_metadata = model.graph.precompute_training_root_lp(costs)
    result = update_loop(model, costs, pools, case, output)
    if result['optimizer_steps'] != CONTINUATION_STEPS:
        raise ControlIncomplete('The continuation did not reach its fixed final update')
    result.update(control_complete=True, usable_checkpoint_saved=True,
        pool_metadata=pool_metadata, root_lp_training=root_metadata,
        config=asdict(model.config), training_only=True)
    model.training_metadata = dict(model.training_metadata,
        fixed_2000_continuation=result)
    nd_training.save_nd_complete(model, output / 'continued-predictor.pt')
    return result


def run(args):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    ctx = common.get_context(args.screen_source, args.screen_output)
    case = common.make_train_case(ctx)
    source = Path(ctx['plan']['source_directory']).resolve()
    fitting.load_api(source, 'neural_diving')
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; no fallback')
    current = fitting.runtime(args.device)
    probe = None
    if args.device == 'cuda':
        probe = fitting.validate_probe(args.probe, ctx, current, gpu=True, method='neural_diving')
    elif not args.cpu_resource_note.strip():
        raise ValueError('Explicit CPU resource rationale is required')
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh control directory; all previous attempts are preserved')
    output.mkdir(parents=True, exist_ok=True)
    core = checkpoint = original = diag = None
    if args.mode == 'continue2000':
        if args.core_fit is None:
            raise ValueError('Continuation requires a sealed core fit')
        core, checkpoint, original = validate_core(case, ctx, args.core_fit)
        diag = validate_gated_core_report(core)
    binding = dict(mode=args.mode, training_only=True, validation_read=False,
        confirmation_read=False, training_sha256=case.dataset_fingerprint,
        train_count=case.n_train, target_K=int(case.affine['rank']),
        parent_bindings=common.evidence_bindings(ctx),
        package_manifest_sha256=common.verify_package(),
        context_provider_sha256=common.sha(common.__file__),
        source_file_sha256=common.sha(__file__),
        native_source_manifest_sha256=common.V4_MANIFEST,
        frozen_source_directory=str(source), runtime=current, resource_probe=probe,
        maximum_seconds=args.max_seconds, original_core=original,
        diagnostic_evidence='Embedded actual gate records bound by the core completion seal',
        cpu_resource_note=args.cpu_resource_note,
        protocol='Frozen v4 core grid restricted by actual1200 eligibility gates; external validation64; separate final2000 continuation')
    common.write(output / 'started.json', dict(binding, started_utc=common.stamp()))
    start = cpu = None
    try:
        with fitting.gpu_exclusive_lock(args.device, current), threadpool_limits(limits=1):
            if args.device == 'cuda':
                free, total = map(int, torch.cuda.mem_get_info())
                allowance = int(probe['scheduling']['per_job_memory_limit_bytes'])
                if allowance + 2 * 1024**3 > free:
                    raise MemoryError('Current GPU free memory is below the verified headroom')
                torch.cuda.set_per_process_memory_fraction(allowance / total)
            fitting.synchronize(args.device)
            start, cpu = time.perf_counter(), time.process_time()
            with deadline(args.max_seconds):
                result = (diagnostic(case, args.device, output) if args.mode == 'diagnostic'
                    else continuation(case, checkpoint, args.device, output))
                fitting.synchronize(args.device)
            elapsed, cpu_elapsed = time.perf_counter()-start, time.process_time()-cpu
        if original is not None:
            if (common.sha(checkpoint) != original['core_weights_sha256'] or
                common.sha(Path(args.core_fit)/'complete.json') != original['core_complete_sha256']):
                raise ValueError('Original selected core artifacts changed')
        report = dict(binding, **result, schema='absolute-nd-control-v1',
            learning_wall_seconds=elapsed, learning_cpu_seconds=cpu_elapsed,
            learning_time_definition='All method-specific graph/pool/root-feature preparation, optimizer work, training diagnostics and final checkpoint serialization. Shared data load and oracle labels excluded.',
            completed_utc=common.stamp())
        if core is not None:
            report['learning_accounting'] = accounting(core, diag, elapsed)
        common.write(output / 'control.json', report)
        common.write(output / 'complete.json', dict(binding,
            schema='absolute-nd-control-complete-v1', control_complete=True,
            artifact_sha256=fitting.artifact_hashes(output), completed_utc=common.stamp()))
        return report
    except BaseException as error:
        common.write(output / 'incomplete.json', dict(binding, control_complete=False,
            usable_for_evaluation=False, error=repr(error), traceback=traceback.format_exc(),
            wall_seconds=None if start is None else time.perf_counter()-start,
            cpu_seconds=None if cpu is None else time.process_time()-cpu,
            retry_requires_new_directory=True, interrupted_utc=common.stamp()))
        raise


def load_continued(case, ctx, folder):
    """Evaluation adapter: return the frozen NeuralLearner around final2000 weights."""
    folder = Path(folder).resolve()
    marker = common.load(folder / 'complete.json')
    if (marker.get('schema') != 'absolute-nd-control-complete-v1' or
        marker.get('mode') != 'continue2000' or not marker.get('control_complete') or
        marker.get('training_sha256') != case.dataset_fingerprint or
        marker.get('target_K') != int(case.affine['rank']) or
        marker.get('parent_bindings') != common.evidence_bindings(ctx) or
        marker.get('source_file_sha256') != common.sha(__file__) or
        marker.get('package_manifest_sha256') != common.verify_package()):
        raise ValueError('Continuation is incomplete or belongs to other case/source')
    validate_artifacts(folder, marker)
    report = common.load(folder / 'control.json')
    if report.get('optimizer_steps') != 2000 or report.get('final_checkpoint_step') != 2000:
        raise ControlIncomplete('Only update2000 weights are evaluable')
    api = fitting.load_api(marker['frozen_source_directory'], 'neural_diving')
    model = load_selected_weights(case, folder / 'continued-predictor.pt', 'cpu')
    metadata = dict(method='neural_diving', training_hash=case.dataset_fingerprint,
        learning_time_seconds=report['learning_accounting']['continuation_cumulative_wall_seconds'],
        fixed_continuation=report, target_K=int(case.affine['rank']))
    return api.NeuralLearner(case, 'neural_diving', model, metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=('diagnostic', 'continue2000'))
    p.add_argument('--screen-source', type=Path, required=True)
    p.add_argument('--screen-output', type=Path, required=True)
    p.add_argument('--device', required=True, choices=('cpu', 'cuda'))
    p.add_argument('--probe', type=Path)
    p.add_argument('--cpu-resource-note', default='')
    p.add_argument('--core-fit', type=Path)
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
