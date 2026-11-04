"""Immutable fits on the shared canonical bank, with original ND sanity gates.

No fitting or CUDA allocation occurs on import. Every invocation uses a fresh
output directory. Validation64 is separate from training512 and confirmation.
"""
from __future__ import annotations
import argparse
import contextlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import bundle_common as common

METHODS = ('mlopt', 'neural_diving', 'preference_learning')
FIT_METHODS = ('preference_learning',)
HERE = Path(__file__).resolve().parent


def verify_source(source):
    source = Path(source).resolve()
    if common.sha(source / 'bundle_manifest.json') != common.V4_MANIFEST:
        raise ValueError('Unchanged frozen v4 source is required')
    for item in common.load(source / 'bundle_manifest.json')['files']:
        path = (source / item['path']).resolve()
        if not path.is_relative_to(source) or common.sha(path) != item['sha256']:
            raise ValueError('Frozen learner file changed: ' + item['path'])
    return source


def load_api(source, method):
    if method not in METHODS:
        raise ValueError('Unknown baseline')
    source = verify_source(source)
    for folder in (source / 'framework', source / 'baselines' / 'neural', source):
        sys.path.insert(0, str(folder))
    # MLOPT's pickle retains this stable module name, used again during load.
    if method == 'mlopt':
        name, path = 'absolute_v4_mlopt_api', source / 'baselines/mlopt/api.py'
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    else:
        module = importlib.import_module('neural_api')
        path = source / 'baselines/neural/neural_api.py'
    if Path(module.__file__).resolve() != path:
        raise ValueError('A different learner module is already imported')
    for name in ('adapter', 'mlopt_baseline', 'cases'):
        loaded = importlib.import_module(name)
        if Path(loaded.__file__).resolve() != source / 'framework' / (name + '.py'):
            raise ValueError('Wrong framework module: ' + name)
    for name in ('neural_support', 'nd_training', 'li_training', 'li_scaling',
                 'preference_baseline', 'neural_diving_baseline'):
        loaded = sys.modules.get(name)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
            raise ValueError('A neural module escaped the frozen source: ' + name)
    return module


def runtime(device):
    import torch
    from importlib.metadata import version
    cpu_model = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu_model = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith('model name')), cpu_model)
    gpu = torch.cuda.get_device_properties(0) if device == 'cuda' and torch.cuda.is_available() else None
    gpu_uuid = None if gpu is None else getattr(gpu, 'uuid', None)
    return dict(hostname=platform.node(), platform=platform.platform(), python=sys.version,
        torch=str(torch.__version__), torch_cuda=torch.version.cuda,
        device=device, cuda_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if device == 'cuda' and torch.cuda.is_available() else None,
        gpu_uuid=None if gpu_uuid is None else str(gpu_uuid),
        packages={name: version(name) for name in ('numpy', 'scipy', 'highspy', 'scikit-learn')},
        cpu_threads=1, cpu_model=cpu_model, available_cpu_count=os.cpu_count(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)


def validate_probe(path, ctx, current, *, gpu=False, li_pairs=False, method=None):
    if path is None:
        raise ValueError('A completed current-bank resource probe is required')
    path = Path(path).resolve()
    report = common.load(path)
    if (report.get('schema') != 'absolute-training-resource-probe-v1' or report.get('success') is not True or
        report.get('training_sha256') != common.TRAINING_HASH or report.get('train_count') != 512 or
        report.get('target_K') != 19 or report.get('parent_bindings') != common.evidence_bindings(ctx) or
        report.get('source_file_sha256') != common.sha(HERE / 'probe.py') or
        report.get('package_manifest_sha256') != common.verify_package() or
        report.get('fit_performed') is not False or report.get('usable_model_saved') is not False):
        raise ValueError('Resource probe is incomplete or belongs to different data/source')
    if gpu:
        previous = report['runtime']
        for name in ('device', 'gpu_name', 'gpu_uuid', 'torch', 'torch_cuda'):
            if previous.get(name) != current.get(name):
                raise ValueError('GPU probe runtime identity differs: ' + name)
        if current.get('device') != 'cuda' or not current.get('gpu_uuid'):
            raise ValueError('A verified current GPU identity is required')
        scheduling = report.get('scheduling', {})
        if scheduling.get('eligible_for_scheduling') is not True:
            raise ValueError('Probe did not establish current-bank GPU headroom')
        if method not in ('neural_diving', 'preference_learning'):
            raise ValueError('This package provides CUDA resource gates for ND/Li; use explicit CPU MLOPT')
        if method not in report.get('resource_methods', []) or not report.get(method, {}).get('success'):
            raise ValueError('The requested method current-bank resource probe must complete')
    if li_pairs:
        estimate = report.get('li_pair_estimate', {})
        if (estimate.get('fixed_training_indices') != [0, 128, 256, 384] or
            estimate.get('training_only') is not True or estimate.get('reused_for_fit') is not False or
            estimate.get('cache_saved') is not False or estimate.get('failures_included_in_timing') is not True or
            len(estimate.get('samples', [])) != estimate.get('sampled_pairs') or not estimate.get('sampled_pairs')):
            raise ValueError('A complete training-only Li pair-cost estimate is required')
    return dict(path=str(path), sha256=common.sha(path), scheduling=report.get('scheduling'),
                li_pair_estimate=report.get('li_pair_estimate'))


@contextlib.contextmanager
def gpu_exclusive_lock(device, current):
    if device != 'cuda':
        yield
        return
    if os.name != 'posix':
        raise RuntimeError('Formal CUDA fitting requires the Linux single-fit lock')
    import fcntl
    safe_uuid = ''.join(c for c in current['gpu_uuid'] if c.isalnum() or c in '-_')
    if not safe_uuid:
        raise ValueError('Missing device UUID for the exclusive fit lock')
    path = Path('/tmp') / ('uc-absolute-fit-' + safe_uuid + '.lock')
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another controlled fit already uses this GPU') from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def synchronize(device):
    if device == 'cuda':
        import torch
        torch.cuda.synchronize()


def validate_case_contract(case, validation):
    if (case.n_train != 512 or validation.n_train != 64 or
        case.dataset_fingerprint != common.TRAINING_HASH or
        case.affine != {'rank': 19} or len(case.commitment_indices) != 180 or
        case.train_features.shape != (512, 15) or validation.train_features.shape != (64, 15)):
        raise ValueError('The frozen 512/64/15-feature/K19 case contract changed')
    if case is validation or case.dataset_fingerprint == validation.dataset_fingerprint:
        raise ValueError('Validation must be a separate frozen bank')


def artifact_hashes(folder):
    folder = Path(folder).resolve()
    result = {}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise ValueError('Fit artifacts cannot contain symlinks')
        if path.is_file() and path.name != 'complete.json':
            if not path.resolve().is_relative_to(folder):
                raise ValueError('Fit artifact escaped its directory')
            result[path.relative_to(folder).as_posix()] = common.sha(path)
    return result


def phase_seconds(method, metadata, total):
    """Only expose actual frozen instrumentation, never impute missing phases."""
    if method == 'mlopt':
        phases = dict(strategy_extraction_and_decoders=float(metadata['bank']['offline_decoder_seconds']),
            network_grid_and_validation=sum(float(t['seconds']) for t in metadata['validation_trials']))
        residual_name = 'pruning_and_other_bookkeeping'
    else:
        raw = metadata['raw_training_metadata']
        if method == 'neural_diving':
            phases = {k: float(raw[k]) for k in ('discarded_gate_seconds', 'feature_preparation_seconds', 'epoch_selection_seconds', 'final_refit_seconds')}
            residual_name = 'other_readiness_and_bookkeeping'
        else:
            phases = {k: float(raw[k]) for k in ('development_selection_seconds', 'final_bank_preparation_seconds', 'final_optimizer_search_seconds')}
            residual_name = 'other_readiness_and_bookkeeping'
    residual = total - sum(phases.values())
    if residual < -1e-4:
        raise ValueError('Frozen learning phase clocks overlap or exceed total')
    phases[residual_name] = max(0., residual)
    return phases


def run_fit(args):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    ctx = common.get_context(args.screen_source, args.screen_output)
    case = common.make_train_case(ctx)
    validation = common.load_validation_case(ctx, args.validation)
    validate_case_contract(case, validation)
    source = Path(ctx['plan']['source_directory']).resolve()
    api = load_api(source, args.method)
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; no silent CPU fallback')
    current_runtime = runtime(args.device)
    probes = {}
    if args.device == 'cuda':
        probes['gpu'] = validate_probe(getattr(args, 'probe', None), ctx, current_runtime, gpu=True, method=args.method)
    elif args.method in ('neural_diving', 'preference_learning') and not getattr(args, 'cpu_resource_note', '').strip():
        raise ValueError('Explicit CPU neural resource rationale is required; no silent fallback')
    if args.method == 'preference_learning':
        pair_path = getattr(args, 'li_pair_probe', None) or getattr(args, 'probe', None)
        probes['li_pairs'] = validate_probe(pair_path, ctx, current_runtime, li_pairs=True)
    folder = Path(args.output).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Preserve previous fit attempts; use a fresh output folder')
    folder.mkdir(parents=True, exist_ok=True)
    validation_folder = Path(args.validation).resolve()
    binding = dict(method=args.method, n_train=512, validation_count=64,
        validation_seed=89001, baseline_seed=42001, target_K=19,
        training_sha256=case.dataset_fingerprint, validation_sha256=validation.dataset_fingerprint,
        validation_seal_sha256=common.sha(validation_folder / 'completion_seal.json'),
        validation_directory=str(validation_folder), frozen_source_directory=str(source),
        native_source_manifest_sha256=common.V4_MANIFEST,
        source_file_sha256=common.sha(__file__), package_manifest_sha256=common.verify_package(),
        parent_bindings=common.evidence_bindings(ctx), runtime=current_runtime,
        probes=probes, controlled_simultaneous_fits_per_gpu=1,
        cpu_resource_note=getattr(args, 'cpu_resource_note', None),
        heldout_confirmation_used=False, original_grid_unchanged=True)
    binding['nd_original_sanity_gates_enforced'] = args.method == 'neural_diving'
    if args.method == 'neural_diving':
        binding['nd_gate_adapter_sha256'] = common.sha(HERE / 'nd_gate_training.py')
        binding['nd_controls_sha256'] = common.sha(HERE / 'nd_controls.py')
    if args.method == 'preference_learning':
        from tie_api import policy_binding
        binding['li_tie_policy'] = policy_binding()
    common.write(folder / 'started.json', dict(binding, started_utc=common.stamp()))
    start = cpu = None
    try:
        synchronize(args.device)
        start, cpu = time.perf_counter(), time.process_time()
        with gpu_exclusive_lock(args.device, current_runtime), threadpool_limits(limits=1):
            if args.device == 'cuda':
                free, total = map(int, torch.cuda.mem_get_info())
                allowance = int(probes['gpu']['scheduling']['per_job_memory_limit_bytes'])
                if allowance + 2 * 1024 ** 3 > free:
                    raise MemoryError('Current free GPU memory is below the probed single-fit headroom')
                torch.cuda.set_per_process_memory_fraction(allowance / total)
            if args.method == 'mlopt':
                model = api.fit(case, validation, config={'seed': 42001, 'device': args.device})
                internal = float(model.manifest['fit_seconds'])
                original_metadata = model.manifest
                model.save(folder / 'model.pkl')
            elif args.method == 'neural_diving':
                from nd_gate_training import fit_nd_with_gate
                model = fit_nd_with_gate(api, case, validation, folder / 'model', args.device)
                internal = float(model.metadata['learning_time_seconds'])
                original_metadata = model.metadata
            else:
                # Li retains every frozen architecture, grid and schedule.
                from tie_api import fit_tie
                model = fit_tie(api, case, validation, folder / 'model', args.device)
                internal = float(model.metadata['learning_time_seconds'])
                original_metadata = model.metadata
        synchronize(args.device)
        outer, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu
        if not (0 <= internal <= outer + 1e-6):
            raise ValueError('Invalid method learning-time boundary')
        report = dict(binding, schema='absolute-baseline-fit-v1',
            learning_wall_seconds=internal, learning_cpu_seconds=cpu_seconds,
            outer_fit_wall_seconds=outer, outer_fit_cpu_seconds=cpu_seconds,
            phase_seconds=phase_seconds(args.method, original_metadata, internal),
            learning_time_definition='All method-specific feature/pair preparation, validation selection, final fitting and readiness checks, including both original ND sanity gates when applicable; shared oracle generation is separate.',
            learning_cpu_scope='Outer full API plus checkpoint process CPU; not an independently instrumented inner learning span.',
            outer_fit_definition='Complete API call including wrapper checks and artifact serialization; imports and shared-bank load are outside this clock.',
            original_metadata=original_metadata, completed_utc=common.stamp(),
            evaluation_variants=(['native', 'matched_K19'] if args.method != 'neural_diving' else ['matched_K19']),
            native_nd_acceptance_ensemble_trained=False,
            native_and_matched_reuse_same_fit=True)
        common.write(folder / 'fit.json', report)
        common.write(folder / 'complete.json', dict(binding,
            schema='absolute-baseline-fit-complete-v1', artifact_sha256=artifact_hashes(folder),
            completed_utc=common.stamp()))
        print(json.dumps(dict(method=args.method, complete=True, learning_wall_seconds=internal,
                              outer_fit_wall_seconds=outer)), flush=True)
        return report
    except BaseException as error:
        if not (folder / 'failure.json').exists():
            common.write(folder / 'failure.json', dict(binding, error=repr(error), traceback=traceback.format_exc(),
                wall_seconds=None if start is None else time.perf_counter() - start,
                cpu_seconds=None if cpu is None else time.process_time() - cpu,
                completed_fit=False, retry_requires_new_directory=True, failed_utc=common.stamp()))
        raise


def load_method(case, method, folder):
    """Load only a complete, source/data-bound fit; return the unchanged API object."""
    if method not in METHODS:
        raise ValueError('Unknown baseline')
    folder = Path(folder).resolve()
    marker = common.load(folder / 'complete.json')
    if (marker.get('schema') != 'absolute-baseline-fit-complete-v1' or marker['method'] != method or
        marker['training_sha256'] != case.dataset_fingerprint or marker['target_K'] != 19 or
        case.affine != {'rank': 19} or marker['native_source_manifest_sha256'] != common.V4_MANIFEST or
        marker['source_file_sha256'] != common.sha(__file__) or
        marker['package_manifest_sha256'] != common.verify_package()):
        raise ValueError('Fit source, training or K binding changed')
    if method == 'neural_diving' and (
        marker.get('nd_original_sanity_gates_enforced') is not True or
        marker.get('nd_gate_adapter_sha256') != common.sha(HERE / 'nd_gate_training.py') or
        marker.get('nd_controls_sha256') != common.sha(HERE / 'nd_controls.py') or
        'fit.json' not in marker.get('artifact_sha256', {}) or
        'model/complete-predictor.pt' not in marker.get('artifact_sha256', {})):
        raise ValueError('ND fit does not bind the actual canonical eligibility-gate protocol')
    for relative, digest in marker['artifact_sha256'].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or path.is_symlink() or common.sha(path) != digest:
            raise ValueError('Saved fit artifact changed: ' + relative)
    if method == 'neural_diving':
        from nd_controls import validate_gated_core_report
        validate_gated_core_report(common.load(folder / 'fit.json'))
    api = load_api(marker['frozen_source_directory'], method)
    if method == 'mlopt':
        model = api.Baseline.load(folder / 'model.pkl')
        if model.manifest['training_sha256'] != case.dataset_fingerprint:
            raise ValueError('MLOPT checkpoint training identity differs')
        return model
    if method == 'preference_learning':
        from tie_api import load_tie, policy_binding
        if marker.get('li_tie_policy') != policy_binding():
            raise ValueError('Li fit is not bound to the reviewed tie policy')
        return load_tie(api, case, folder / 'model')
    return api.load(case, method, folder / 'model')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-source', type=Path, required=True)
    parser.add_argument('--screen-output', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--method', choices=FIT_METHODS, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--probe', type=Path, help='Completed same-bank, same-GPU resource probe for CUDA fitting')
    parser.add_argument('--li-pair-probe', type=Path, help='Completed training-only Li pair-cost probe, or use --probe from all mode')
    parser.add_argument('--cpu-resource-note', default='', help='Required explicit rationale for neural CPU fitting')
    parser.add_argument('--output', type=Path, required=True)
    run_fit(parser.parse_args())


if __name__ == '__main__':
    main()
