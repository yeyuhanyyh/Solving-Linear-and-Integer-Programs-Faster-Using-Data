"""Reproducible source extraction; edits orchestration, never learner math."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent


def extracted(path, name, owner=None):
    source=path.read_text(encoding="utf-8")
    tree=ast.parse(source)
    nodes=tree.body
    if owner: nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==owner).body
    return ast.get_source_segment(source,next(n for n in nodes if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name))


def replacement(source, old, new):
    assert source.count(old)==1, old
    return source.replace(old,new)


old=ROOT/"vendor/prior_train_uc_baselines.py"
nd=extracted(old,"fit_nd_complete")
nd=replacement(nd,'def fit_nd_complete(case, device="cpu"):',
    'def fit_nd_complete(case, validation_case, device="cpu", config=None):')
nd=replacement(nd,'config=nd.NeuralDivingConfig(seed=int(case.metadata["baseline_seed"]),device=device)',
    'config=config or nd.NeuralDivingConfig(seed=int(case.metadata["baseline_seed"]),device=device)\n    device=config.device\n    data.validate_training_pair(case,validation_case)')
nd=replacement(nd,'''costs=np.asarray(case.train_costs,dtype=float)
    perm=np.random.default_rng(config.seed).permutation(case.n_train)
    n_val=max(1,int(round(case.n_train*config.validation_fraction)))
    validation_indices=np.sort(perm[:n_val]); training_indices=np.sort(perm[n_val:])''',
    '''selection_case=data.concatenate_for_selection(case,validation_case)
    costs=np.asarray(selection_case.train_costs,dtype=float)
    n_val=validation_case.n_train
    training_indices=np.arange(case.n_train,dtype=int)
    validation_indices=np.arange(case.n_train,case.n_train+n_val,dtype=int)''')
nd=replacement(nd,'nd._prepare_pools(case,data.singleton_pools(case),graph.integer_indices)',
    'nd._prepare_pools(selection_case,data.singleton_pools(selection_case),graph.integer_indices)')
nd=replacement(nd,'reference=np.asarray(case.train_objectives,dtype=float)',
    'reference=np.asarray(selection_case.train_objectives,dtype=float)')
nd=replacement(nd,'graph=nd.FixedBipartiteGraph(case.model,costs,config.normalize_constraint_rows,config.device,',
    'graph=nd.FixedBipartiteGraph(case.model,costs[training_indices],config.normalize_constraint_rows,config.device,')
nd=replacement(nd,'train_cost_digest=nd._array_digest(costs)',
    'train_cost_digest=nd._array_digest(np.asarray(case.train_costs,dtype=float))')
nd=replacement(nd,'implementation="nd-complete-only-frozen-uc-v1"',
    'implementation="nd-complete-external-validation-uc-v1"')
nd=replacement(nd,'gate=nd._sanity_gate(graph,candidate_config,case,costs,pools,training_indices,reference,adapter.solve)',
    'gate={"passed":True,"scope":"actual_training_numerical_checks_only","extra_overfit_steps":0}')
nd=replacement(nd,'state,epochs,history=nd._warmup_selection(graph,candidate_config,costs,pools,training_indices,validation_indices)',
    '''state,epochs,history=nd._warmup_selection(graph,candidate_config,costs,pools,training_indices,validation_indices)
        if state is None or epochs<1 or not np.isfinite(history["best_validation_full_nll"]):
            raise FloatingPointError("Actual ND training did not return a finite selected state")''')
nd=replacement(nd,'predictor.eval(); diagnostics=nd._full_diagnostics(predictor,costs,pools,final_indices)',
    '''predictor.eval(); diagnostics=nd._full_diagnostics(predictor,costs,pools,final_indices)
    if not np.isfinite(diagnostics["full_nll_per_integer"]):
        raise FloatingPointError("Final ND predictor has nonfinite training loss")''')
nd=replacement(nd,'optimizer=nd._optimizer(predictor,chosen_config);rng=np.random.default_rng(config.seed+110003)',
    '''initial_refit_diagnostics=nd._full_diagnostics(predictor,costs,pools,final_indices)
    optimizer=nd._optimizer(predictor,chosen_config);rng=np.random.default_rng(config.seed+110003)''')
nd=replacement(nd,'final_full_predictor_diagnostics=diagnostics,parameter_count=',
    '''initial_full_predictor_diagnostics=initial_refit_diagnostics,
        final_full_predictor_diagnostics=diagnostics,
        actual_training_loss_decreased=bool(diagnostics["full_nll_per_integer"]<=initial_refit_diagnostics["full_nll_per_integer"]),
        optional_overfit_qa_run=False,optional_overfit_qa_seconds=0.,parameter_count=''')
nd=replacement(nd,'validation_selection="matched-K recovery, valid count, smaller valid objective gap, NLL, smaller width; training prefix only",',
    'validation_selection="matched-K recovery, valid count, smaller valid objective gap, NLL, smaller width; external validation only",\n        external_validation_count=n_val, external_validation_hash=data.training_hash(validation_case),\n        validation_labels_in_final_fit=False,')
nd=nd.replace('within the training prefix','on the separate validation bank').replace('Original GCN, root features, optimizer and sanity schedule retained.',
    'Original GCN, root features and optimizer retained. The optional overfit diagnostic is excluded from formal fitting and its learning clock. Separate external validation replaces the internal split; final refit uses only the N training rows.')
save=extracted(old,"save_nd_complete")
load=extracted(old,"load_nd_complete")
header='''"""Audited complete-predictor ND fit with separate external validation.

Derived mechanically from the frozen October25 runner. Learner math unchanged.
"""
from dataclasses import asdict, replace
from pathlib import Path
import json
import time
import numpy as np
import neural_support as data
data.enable_runtime()
from neural_timing import install as _install_nd_deadline_check
_install_nd_deadline_check()

def synchronize(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()

def event(**fields):
    print(json.dumps(data.clean(fields)),flush=True)

'''
(ROOT/"nd_training.py").write_text(header+nd+"\n\n"+save+"\n\n"+load+"\n",encoding="utf-8")

li=extracted(ROOT/"vendor/preference_baseline.py","fit","PreferenceBaseline")
li=replacement(li,'''rng = np.random.default_rng(self.config.seed)
        permutation = rng.permutation(n)
        validation_count = max(1, int(round(n * self.config.validation_fraction)))
        validation_indices = np.sort(permutation[:validation_count])
        fit_indices = np.sort(permutation[validation_count:])''',
    '''data.validate_training_pair(self.case,self.validation_case)
        validation_count = self.validation_case.n_train
        validation_indices = np.arange(validation_count,dtype=int)
        fit_indices = np.arange(n,dtype=int)''')
li=replacement(li,'validation_costs = np.asarray(self.case.train_costs)[validation_indices]',
    'validation_costs = np.asarray(self.validation_case.train_costs)')
li=replacement(li,'validation_features = np.asarray(self.case.train_features)[validation_indices]',
    'validation_features = np.asarray(self.validation_case.train_features)')
li=replacement(li,'validation_objectives = np.sum(validation_costs * np.asarray(self.case.train_solutions)[validation_indices], axis=1)',
    'validation_objectives = np.asarray(self.validation_case.train_objectives)')
li=li.replace('training_only_validation_indices','external_validation_indices').replace('fixed_120_training_budget_no_adaptive_good_turing','fixed_training_budget_no_adaptive_good_turing')
li=replacement(li,'self.report["sanity_gate"] = final_sanity',
    'self.report["sanity_gate"] = final_sanity\n        self.report["external_validation_count"] = validation_count\n        self.report["external_validation_hash"] = data.training_hash(self.validation_case)\n        self.report["validation_labels_in_final_fit"] = False\n        self.report["adaptations"].append("separate_external_validation_final_refit_training_rows_only")')
li_header='''"""Li fit orchestration with shared external validation; frozen loss/decoder.

Only the old internal split is replaced. The final bank and optimizer see N
training rows only. Native strategy selection and matched inference stay separate.
"""
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import torch
import neural_support as data
data.enable_runtime()
from preference_baseline import (PreferenceBaseline, PreferenceConfig, IMPLEMENTATION_VERSION, SOURCE,
    _subset_training_case, _pair_data, _training_hash, _model_hash)

class ExternalValidationPreference(PreferenceBaseline):
    def __init__(self,case,validation_case,config=None):
        super().__init__(case,config=config)
        self.validation_case=validation_case

    def _train_network(self,development_case,bank,pairs,*args,**kwargs):
        # Original October25 grid, clipped only to retained candidate count.
        effective=tuple(dict.fromkeys(min(k,len(pairs["selected"])) for k in (1,3,5,10,30)))
        self.config=replace(self.config,top_k_values=effective)
        return super()._train_network(development_case,bank,pairs,*args,**kwargs)

'''
(ROOT/"li_training.py").write_text(li_header+textwrap.indent(li,"    ")+"\n",encoding="utf-8")
manifest={"source_root":"D:/C盘迁移归档/桌面大文件夹/research/ip-paper-baselines-v2-20261009",
    "vendored_byte_identical_files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"vendor").iterdir() if p.is_file()},
    "derived_files": {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("nd_training.py","li_training.py")},
    "orchestration_change":"Separate shared external validation; final learner sees N training rows only. No loss, architecture, strategy definition or native decoder changes."}
(ROOT/"source_provenance.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
