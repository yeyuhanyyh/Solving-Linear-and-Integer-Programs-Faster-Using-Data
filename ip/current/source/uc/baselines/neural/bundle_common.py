"""Bindings for the separately sealed canonical common representation.

The training bank and K were fixed by the earlier absolute-cost experiment.
This module never reads discovery/confirmation objectives or fits a learner.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V4_MANIFEST = 'e693ee1f3be695b6650dd35dac3801b8e7ee7ed17c0fb220d06a052162d40680'
PARENT_PLAN = '866cd97700d360d38b4ac9ea6249d1a7d3ea3de4aa144cbc3be1ad029d99b841'
TRAINING_HASH = None  # Assigned only after checking the sealed canonical contract.
MODEL_HASH = None
ANCHOR_ORIGINAL_HASH = '42267dfab7393de06b4dc2be6df1f2f549b9b2ce2bfef80fc91561e82b12fbcf'
N_TRAIN, N_VALIDATION, N_CONFIRMATION, TARGET_K, COMMON_D = 512, 64, 256, 19, 180
TRAIN_SEED, DISCOVERY_SEED, CONFIRMATION_SEED, VALIDATION_SEED = 86001, 87001, 88001, 89001
BASELINE_SEED = 42001


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    # Keep importing NumPy/scientific dependencies outside this low-level helper.
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float):
        import math
        if not math.isfinite(value):
            return None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def package_hashes():
    """Hash the separately packaged extension, never the parent source tree."""
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py'))}


def verify_package():
    manifest_path = HERE / 'package_manifest.json'
    if not manifest_path.exists():
        raise ValueError('Build and verify the sealed extension package before running it')
    manifest = load(manifest_path)
    if manifest.get('schema') != 'absolute-full-baselines-package-v1':
        raise ValueError('Wrong extension package schema')
    if manifest.get('execution_stage') != 'canonical-full':
        raise ValueError('This runner requires its separately sealed canonical package')
    required = {'bundle_common.py', 'validation.py', 'fit_bundle.py', 'probe.py',
                'evaluation.py', 'matched_quality.py', 'vendor_gpu_probe.py',
                'structure_core.py', 'matched_candidates.py', 'nd_gate_training.py',
                'nd_controls.py', 'reference_certificate.py', 'tie_bank.py', 'tie_api.py', 'own_strategy_gate.py', 'protocol.json'}
    if not required.issubset(manifest.get('files', {})):
        raise ValueError('Extension manifest does not cover every runtime module')
    for relative, expected in manifest['files'].items():
        path = (HERE / relative).resolve()
        if not path.is_relative_to(HERE) or sha(path) != expected:
            raise ValueError('Extension package file changed: ' + relative)
    if manifest.get('v4_manifest_sha256') != V4_MANIFEST:
        raise ValueError('Wrong frozen learner source')
    return sha(manifest_path)


def fixed_anchor_evidence(ctx, projected_anchor, cost, features):
    """Evaluate the actual old full anchor, never an arbitrary canonical lift."""
    import numpy as np
    canonical = ctx['_canonical_module']
    training = canonical.load_split_data(ctx, 'absolute', 'train')
    if not np.array_equal(projected_anchor, training['solutions'][0]):
        raise ValueError('Projected first-training anchor changed')
    if not np.allclose(cost, canonical.objective(ctx, features), rtol=0., atol=1e-9):
        raise ValueError('Anchor objective does not match the supplied cost features')
    evidence = dict(canonical.fixed_anchor_evidence(ctx, features))
    objective = evidence['objective_recomputed'] if 'objective_recomputed' in evidence else evidence['objective']
    original_hash = evidence['original_anchor_sha256'] if 'original_anchor_sha256' in evidence else evidence['anchor_original_sha256']
    if evidence['valid'] is not True or original_hash != ANCHOR_ORIGINAL_HASH:
        raise ValueError('The actual original complete first-training anchor changed')
    if not np.isfinite(objective) or not np.isclose(float(cost @ projected_anchor), float(objective), rtol=1e-10, atol=1e-7):
        raise ValueError('Projected anchor objective differs from the actual original anchor')
    evidence.update(objective_recomputed=float(objective), original_anchor_sha256=original_hash)
    return evidence


def project_external_cost(ctx, cost, features):
    """Check old coefficients before using one reproducible runtime expression."""
    import numpy as np
    canonical = ctx['_canonical_module']
    projected = np.asarray(canonical.project_cost(ctx, cost), dtype=float)
    expected = np.asarray(canonical.objective(ctx, features), dtype=float)
    if projected.shape != expected.shape or not np.allclose(projected, expected, rtol=0., atol=1e-9):
        raise ValueError('External objective is not the same canonical cost family')
    return expected


def get_context(screen_source, screen_output):
    """Load canonical source and banks only after verifying their exact seals."""
    global TRAINING_HASH, MODEL_HASH
    import numpy as np
    package_manifest = verify_package()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    source, output = Path(screen_source).resolve(), Path(screen_output).resolve()
    protocol = load(HERE / 'protocol.json')
    pins = protocol.get('canonical_binding', {})
    required_pins = {'source_manifest_sha256', 'completion_seal_sha256',
                     'model_sha256', 'training_fingerprint', 'validation_fingerprint'}
    if not required_pins.issubset(pins) or not all(isinstance(pins[k], str) and len(pins[k]) == 64 for k in required_pins):
        raise ValueError('The canonical model, banks and source have not been pinned')
    if sha(source / 'source_manifest.json') != pins['source_manifest_sha256']:
        raise ValueError('Canonical source manifest differs from the suite contract')
    if sha(output / 'completion_seal.json') != pins['completion_seal_sha256']:
        raise ValueError('Canonical data seal differs from the suite contract')
    existing = sys.modules.get('canonical_common')
    if existing is not None and Path(existing.__file__).resolve().parent != source:
        raise ValueError('Use a fresh process for a different canonical source version')
    sys.path.insert(0, str(source))
    canonical = importlib.import_module('canonical_common')
    if Path(canonical.__file__).resolve().parent != source:
        raise ValueError('Wrong canonical module')
    canonical.verify_package(source)
    ctx = dict(canonical.load_context(output, load_runtime=True))
    if ctx['_external_parent_plan_sha256'] != PARENT_PLAN:
        raise ValueError('Canonical model does not descend from the fixed absolute parent')
    if ctx['linear_model'].A.shape != (2544, 2844) or len(ctx['model']['commitment_indices']) != COMMON_D:
        raise ValueError('Wrong canonical common physical representation')
    MODEL_HASH = pins['model_sha256']
    TRAINING_HASH = pins['training_fingerprint']
    if ctx['rt']['cases'].model_hash(ctx['linear_model']) != MODEL_HASH:
        raise ValueError('Canonical physical model changed')
    arrays = canonical.load_split_data(ctx, 'absolute', 'train')
    if arrays['features'].shape != (N_TRAIN, 15) or not np.all(arrays['certified']):
        raise ValueError('Expected all512 projected original-certified training solutions')
    case = canonical.make_training_case(ctx)
    if case.dataset_fingerprint != TRAINING_HASH:
        raise ValueError('Canonical training bank changed')
    ctx.update(_common=canonical, _canonical_module=canonical)
    fixed_anchor_evidence(ctx, arrays['solutions'][0], arrays['costs'][0], arrays['features'][0])
    bindings = dict(parent_plan_sha256=PARENT_PLAN, v4_manifest_sha256=V4_MANIFEST,
        training_array_sha256=TRAINING_HASH, model_sha256=MODEL_HASH,
        anchor_original_sha256=ANCHOR_ORIGINAL_HASH, target_K=TARGET_K,
        canonical_plan_sha256=ctx['plan_sha256'],
        canonical_source_manifest_sha256=pins['source_manifest_sha256'],
        canonical_completion_seal_sha256=pins['completion_seal_sha256'],
        validation_fingerprint=pins['validation_fingerprint'],
        extension_manifest_sha256=package_manifest)
    ctx.update(_common=canonical, _canonical_module=canonical,
        _bindings=bindings, _screen_source=source, _screen_output=output)
    ctx['_project_external_cost'] = lambda cost, features: project_external_cost(ctx, cost, features)
    ctx['_project_external_solution'] = lambda point: canonical.project_point(ctx, point)
    ctx['_fixed_anchor_evidence'] = lambda anchor, cost, features: fixed_anchor_evidence(ctx, anchor, cost, features)
    return ctx


def make_train_case(ctx):
    """A new view with fixed K, leaving parent arrays/artifacts unmodified."""
    case = ctx['_canonical_module'].make_training_case(ctx)
    if case.dataset_fingerprint != TRAINING_HASH:
        raise ValueError('Training identity changed')
    case.affine = {'rank': TARGET_K}
    case.metadata = dict(case.metadata, baseline_seed=BASELINE_SEED,
        full_baseline_extension=True, validation_seed=VALIDATION_SEED,
        heldout_information_used=False)
    return case


def load_validation_case(ctx, folder):
    """Read the projected fresh64 bank; no new oracle or fitting is invoked."""
    if Path(folder).resolve() != Path(ctx['_screen_output']).resolve():
        raise ValueError('Validation must come from the same pinned canonical completion seal')
    case = ctx['_canonical_module'].make_validation_case(ctx)
    if case.n_train != N_VALIDATION or case.dataset_fingerprint != ctx['_bindings']['validation_fingerprint']:
        raise ValueError('Canonical fresh64 validation bank changed')
    from validation import verify_features
    verify_features(case.train_features)
    if case.train_costs.shape != (N_VALIDATION, ctx['linear_model'].A.shape[1]):
        raise ValueError('Wrong canonical validation dimensions')
    case.affine = {'rank': TARGET_K}
    return case


def evidence_bindings(ctx):
    return dict(ctx['_bindings'])
