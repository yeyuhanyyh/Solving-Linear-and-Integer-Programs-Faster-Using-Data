"""Explicit Li v2 API: only certified representative tie decoding changes.

The inherited grid, pair loss, SetCover, external validation, final optimization
and native strategy selection are unchanged. Fresh directories and a separate
checkpoint version prevent reuse of old-decoder pair caches or checkpoints.
"""
from __future__ import annotations
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
import neural_support as data
import preference_baseline as li
from li_training import ExternalValidationPreference
from li_scaling import FactorizedPreferenceNetwork
from tie_bank import TieBank, VERSION, source_fingerprint

EVIDENCE_FIELDS = ('reduced_certificate', 'representative_tie',
    'reduced_solver_certified', 'reduced_solver_dual_bound', 'original_certified')


class QueryEvidenceBank:
    """One-query proxy retaining only evidence omitted by the original whitelist."""
    def __init__(self, bank):
        self.bank = bank
        self.evidence = {}

    def __getattr__(self, name):
        return getattr(self.bank, name)

    def decode(self, strategy_index, *args, **kwargs):
        decoded = self.bank.decode(strategy_index, *args, **kwargs)
        self.evidence[int(strategy_index)] = {name:deepcopy(decoded[name])
            for name in EVIDENCE_FIELDS if name in decoded}
        return decoded


def policy_binding():
    return dict(version=VERSION, source_sha256=source_fingerprint(),
        base_implementation_version=li.IMPLEMENTATION_VERSION,
        pair_cache_reuse_allowed=False,
        representative_rule='Fixed representative_training_index only; same certified reduced-LP optimum')


def versioned_cache_key(native_key):
    material = dict(policy=policy_binding(), native_cache_key=str(native_key))
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _cache_binding(report):
    output = {}
    for phase in ('development', 'final'):
        entry = report['pair_cache'][phase]
        if entry.get('cache_reused') is not False:
            raise ValueError('Li tie-v2 forbids reuse of prior pair caches')
        output[phase] = dict(native_cache_key=entry['cache_key'],
            versioned_cache_key=versioned_cache_key(entry['cache_key']), cache_reused=False)
    return output


class TieExternalValidationPreference(ExternalValidationPreference):
    """Inherited training; explicit strategy-bank and checkpoint specialization."""
    def _bank(self, case):
        return TieBank(case, self.config)

    def _solve_with(self, bank, selected, network, mean, scale, cost, features,
                    top_k, total_time_limit=30.0):
        # The original routine alone performs ranking, decoding and Eq.(21)
        # selection. This proxy neither changes its returns nor calls a solver.
        proxy = QueryEvidenceBank(bank)
        result = super()._solve_with(proxy, selected, network, mean, scale,
            cost, features, top_k, total_time_limit=total_time_limit)
        for candidate in result.get('candidates', []):
            candidate.update(proxy.evidence.get(candidate.get('strategy_id'), {}))
        result.update(proxy.evidence.get(result.get('strategy_id'), {}))
        return result

    def save(self, output_dir):
        # Same tensors and payload fields as the original save. The version is
        # deliberately distinct so the old native loader cannot silently drop
        # the corrected decoder policy while retaining these trained weights.
        if self.network is None or self.selected is None:
            raise RuntimeError('No fitted preference model to save')
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.report['tie_policy'] = policy_binding()
        self.report['versioned_pair_cache'] = _cache_binding(self.report)
        self.report['decode'] = 'One certified native reduced LP; fixed training representative may resolve an objective tie only after full and reduced feasibility checks'
        data.write_json(destination/'preference-training.json', self.report)
        checkpoint = dict(state_dict={name:tensor.detach().cpu() for name,tensor in self.network.state_dict().items()},
            configuration=asdict(self.config), implementation_version=VERSION,
            base_implementation_version=li.IMPLEMENTATION_VERSION, tie_policy=policy_binding(),
            sanity_passed=bool(self.report.get('sanity_passed', False)),
            pair_cache_key=self.report['pair_cache']['final']['cache_key'],
            versioned_pair_cache_key=self.report['versioned_pair_cache']['final']['versioned_cache_key'],
            training_hash=li._training_hash(self.case), model_hash=li._model_hash(self.case),
            selected_strategy_ids=self.selected.tolist(), strategy_signatures=li._strategy_signatures(self.bank),
            mean=self.mean, scale=self.scale, top_k=self.top_k)
        with (destination/'preference-network.pt').open('wb') as stream:
            torch.save(checkpoint, stream)


def _artifacts(destination):
    destination = Path(destination).resolve()
    output = {}
    for path in sorted(destination.rglob('*')):
        if path.is_symlink():
            raise ValueError('Tie-v2 artifacts cannot contain symlinks')
        if path.is_file() and path.name != 'neural-complete.json':
            output[path.relative_to(destination).as_posix()] = data.sha(path)
    return output


def fit_tie(api, case, validation, output, device):
    """Fresh Li-only fit; return the original NeuralLearner interface."""
    if li.PreferenceNetwork is not FactorizedPreferenceNetwork:
        raise ValueError('Expected the unchanged frozen factorized network')
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Tie-v2 requires a fresh directory; old and failed caches are preserved separately')
    destination.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    config = li.PreferenceConfig(seed=int(case.metadata['baseline_seed']), device=device,
        top_k_values=(1,3,5,10,30))
    guard = dict(method='preference_learning', training_hash=data.training_hash(case),
        validation_hash=data.training_hash(validation), source_sha256=data.implementation_fingerprint(),
        device=device, configuration=asdict(config), tie_policy=policy_binding(),
        fresh_pair_caches_required=True, test_information_used=False)
    data.write_json(destination/'fit_guard.json', guard)
    data.validate_training_pair(case, validation)
    model = TieExternalValidationPreference(case, validation, config=config).fit(output_dir=destination)
    raw = model.metadata
    cache_binding = _cache_binding(raw)
    if raw.get('tie_policy') != policy_binding():
        raise ValueError('Saved fit does not record the exact tie-v2 decoder policy')
    metadata = dict(method='preference_learning', fit_wall_seconds=perf_counter()-started,
        learning_time_seconds=float(raw['fit_seconds']), training_count=case.n_train,
        external_validation_count=validation.n_train, training_hash=data.training_hash(case),
        validation_hash=data.training_hash(validation), target_K=int(case.affine['rank']),
        learning_time_excludes_shared_label_generation=True,
        learning_time_includes='Original pair preparation with all tie checks, external-validation selection, final fitting and readiness checks',
        fit_wall_time_additionally_includes='Wrapper data validation and checkpoint serialization',
        original_algorithm=False, matched_dimension_rule='Exactly D-K common commitment coordinates are fixed',
        matched_inference_is_explicit_adaptation=True, raw_training_metadata=raw,
        tie_policy=policy_binding(), versioned_pair_cache=cache_binding,
        validation_labels_in_final_fit=False, heldout_confirmation_used=False)
    data.write_json(destination/'metadata.json', metadata)
    data.write_json(destination/'neural-complete.json', dict(method='preference_learning',
        schema='li-tie-v2-complete', training_hash=metadata['training_hash'],
        validation_hash=metadata['validation_hash'], source_sha256=guard['source_sha256'],
        tie_policy=policy_binding(), artifact_sha256=_artifacts(destination)))
    return api.NeuralLearner(case, 'preference_learning', model, metadata)


def verify_complete(case, output):
    """Check source, policy and every artifact before any pickle/torch load."""
    destination = Path(output).resolve()
    complete = json.loads((destination/'neural-complete.json').read_text(encoding='utf-8'))
    if (complete.get('schema') != 'li-tie-v2-complete' or
        complete.get('method') != 'preference_learning' or
        complete.get('training_hash') != data.training_hash(case) or
        complete.get('source_sha256') != data.implementation_fingerprint() or
        complete.get('tie_policy') != policy_binding()):
        raise ValueError('Tie-v2 completion belongs to different data, source or decoder policy')
    required = {'preference-network.pt','preference-training.json','metadata.json','fit_guard.json'}
    if not required.issubset(complete.get('artifact_sha256',{})):
        raise ValueError('Tie-v2 completion does not bind all required artifacts')
    for relative, digest in complete['artifact_sha256'].items():
        path = (destination/relative).resolve()
        if not path.is_relative_to(destination) or path.is_symlink() or data.sha(path) != digest:
            raise ValueError('Tie-v2 artifact changed: '+relative)
    metadata = json.loads((destination/'metadata.json').read_text(encoding='utf-8'))
    guard = json.loads((destination/'fit_guard.json').read_text(encoding='utf-8'))
    if (metadata.get('tie_policy') != policy_binding() or guard.get('tie_policy') != policy_binding() or
        metadata.get('training_hash') != complete['training_hash'] or
        metadata.get('validation_hash') != complete.get('validation_hash') or
        guard.get('training_hash') != complete['training_hash'] or
        guard.get('validation_hash') != complete.get('validation_hash') or
        guard.get('source_sha256') != complete['source_sha256']):
        raise ValueError('Tie-v2 source/training/validation bindings disagree')
    if metadata.get('versioned_pair_cache') != _cache_binding(metadata['raw_training_metadata']):
        raise ValueError('Tie-v2 pair cache source version changed')
    return complete, metadata


def load_tie(api, case, output):
    """Explicit TieBank restoration; original checkpoint math and grid unchanged."""
    destination = Path(output).resolve()
    complete, metadata = verify_complete(case, destination)
    with (destination/'preference-network.pt').open('rb') as stream:
        checkpoint = torch.load(stream, map_location='cpu', weights_only=False)
    if (checkpoint.get('implementation_version') != VERSION or
        checkpoint.get('base_implementation_version') != li.IMPLEMENTATION_VERSION or
        checkpoint.get('tie_policy') != policy_binding() or
        not checkpoint.get('sanity_passed',False)):
        raise ValueError('Checkpoint is not a ready tie-v2 preference model')
    if li._training_hash(case) != checkpoint['training_hash'] or li._model_hash(case) != checkpoint['model_hash']:
        raise ValueError('Checkpoint training arrays or physical model changed')
    # Validation is not needed for inference. Passing None creates no bank and
    # performs no fit; its immutable training case supplies all strategies.
    fitted = TieExternalValidationPreference(case, None, config=li.PreferenceConfig(**checkpoint['configuration']))
    li._seed_everything(fitted.config.seed)
    fitted.bank = fitted._bank(li._subset_training_case(case,np.arange(len(case.train_costs))))
    if li._strategy_signatures(fitted.bank) != checkpoint['strategy_signatures']:
        raise ValueError('Checkpoint strategy bank differs from training data')
    native_key = li._pair_cache_key(case,fitted.bank,fitted.config)
    if (native_key != checkpoint['pair_cache_key'] or
        versioned_cache_key(native_key) != checkpoint.get('versioned_pair_cache_key')):
        raise ValueError('Checkpoint pair-cache/source configuration pin changed')
    fitted.selected = np.asarray(checkpoint['selected_strategy_ids'],dtype=int)
    fitted.mean = np.asarray(checkpoint['mean'],dtype=float)
    fitted.scale = np.asarray(checkpoint['scale'],dtype=float)
    fitted.top_k = int(checkpoint['top_k'])
    fitted.network = FactorizedPreferenceNetwork(len(fitted.mean),li._strategy_features(fitted.bank,fitted.selected),fitted.config)
    fitted.network.load_state_dict(checkpoint['state_dict'])
    fitted.network.eval()
    fitted.report = json.loads((destination/'preference-training.json').read_text(encoding='utf-8'))
    if fitted.report.get('tie_policy') != policy_binding() or not fitted.report.get('sanity_passed',False):
        raise ValueError('Training report does not certify this tie-v2 fit')
    return api.NeuralLearner(case,'preference_learning',fitted,metadata)
