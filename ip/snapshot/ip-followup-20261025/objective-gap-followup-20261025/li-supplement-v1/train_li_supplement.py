"""Three fixed G32/n36 Li fits; no test/development bank is opened here."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import contextlib
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
FOLLOWUP=ROOT.parents[1]
GENERIC=FOLLOWUP/'objective-gap-designs-20261025/adapter/generic-binary-v2'
DRIVER=GENERIC/'generic_binary_pilot.py'
DRIVER_SHA='d7ff552f55775f90387444e8cbc6290ee371ca94960f7280f42c762c8768e8c2'
MANIFEST=FOLLOWUP/'objective-gap-designs-20261025/coupled-bundle/scale-extension-g32-v1/short-prefix-n36/sealed-cohort-v1/training-only-manifest.json'
MANIFEST_SHA='e1c5bc638952fa690137ee572e5c7974e14a418aaa7f214b20819f9a06c1f7a0'
PROTOCOL=ROOT/'protocol.json'
OLD_FREEZE=ROOT.parent/'uniform-gross-stream2-training-budget-control/all-five-role-models-freeze.json'
OLD_FREEZE_SHA='b40db9584786810ab1a661ac8d8968f232663bb04bf87f5ff336db5b9b737d26'
_GENERIC=None


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def enable():
    global _GENERIC
    if sha(DRIVER)!=DRIVER_SHA or sha(MANIFEST)!=MANIFEST_SHA or sha(OLD_FREEZE)!=OLD_FREEZE_SHA:
        raise ValueError('Frozen source/training/previous-bank population changed')
    if _GENERIC is None:
        spec=importlib.util.spec_from_file_location('li_supplement_frozen_generic',DRIVER)
        _GENERIC=importlib.util.module_from_spec(spec);sys.modules[spec.name]=_GENERIC;spec.loader.exec_module(_GENERIC)
    _GENERIC.enable_runtime();_GENERIC.bind_signed_cost_map()
    import torch
    import train_uc_baselines as nn
    torch.set_num_threads(1);nn.model_guard=guard
    return _GENERIC
def guard(case,method,bundle=None):
    d=enable()
    return dict(original_training=d.pilot_guard(case,method),supplement_training_source_sha256=sha(__file__),
        protocol_sha256=sha(PROTOCOL),method=method,case_id=case.case_id,n_train=36,D=96,K=32,
        original_shared_trainer_sha256=sha(GENERIC/'shared/train_uc_baselines.py'),
        original_li_implementation_sha256=sha(GENERIC/'runtime/preference_baseline.py'),
        test_information_used_for_selection=False,later_baseline_supplement=True)
def cases():
    d=enable();manifest=d.validate_selection(MANIFEST);result=[]
    for entry in manifest['cases']:
        case=d.load_case(entry['directory'],MANIFEST)
        if (case.n_train,case.model.n_variables,int(case.affine['rank']))!=(36,96,32):raise ValueError('Wrong fixed case')
        result.append(case)
    if len(result)!=3 or {case.seed for case in result}!={1009,1013,1019}:raise ValueError('Require all three seeds')
    return sorted(result,key=lambda case:case.seed)
def write(path,value):enable().write_new(path,value)


def freeze_models(all_cases):
    import train_uc_baselines as nn
    sources={str(path.resolve()):sha(path) for path in (Path(__file__),PROTOCOL,DRIVER,MANIFEST,
        GENERIC/'runtime-seal.json',GENERIC/'GENERIC_BINARY_PROTOCOL.md',OLD_FREEZE)}
    for name,expected in read(GENERIC/'runtime-seal.json')['file_sha256'].items():
        path=GENERIC/name
        if sha(path)!=expected:raise ValueError('Original scientific runtime changed')
        sources[str(path.resolve())]=expected
    entries={}
    manifest={entry['case_id']:entry for entry in read(MANIFEST)['cases']}
    for case in all_cases:
        folder,marker=nn.validate_checkpoint(ROOT,case,'li',GENERIC)
        if marker['training_label_count']!=36 or marker['sanity_gate_passed'] is not True:raise ValueError('Incomplete Li training')
        files={str((folder/'complete.json').resolve()):sha(folder/'complete.json')}
        for name,expected in marker['artifact_sha256'].items():
            path=folder/name
            if sha(path)!=expected:raise ValueError('Li model artifact changed')
            files[str(path.resolve())]=expected
        if not any(Path(path).name=='preference-network.pt' for path in files):raise ValueError('Missing actual preference weights')
        source=manifest[case.case_id]
        entries[case.case_id]=dict(case_id=case.case_id,seed=case.seed,G=32,D=96,K=32,n_train=36,
            training_directory=source['directory'],training_files={str((Path(source['directory'])/name).resolve()):expected for name,expected in source['files'].items()},
            guard=guard(case,'li'),files=files,fit_seconds=marker['fit_seconds'],fit_cpu_seconds=marker['fit_cpu_seconds'])
    result=dict(schema='coupled-knapsack-later-Li-supplement-models-v1',frozen=True,model_count=3,
        frozen_at_utc=utc(),cases=entries,source_sha256=sources,original_stream2_model_freeze_sha256=OLD_FREEZE_SHA,
        chronology='Li training and selection occurred after stream2 cost generation. Li models are frozen before their first stream2 evaluation. This is a later supplemental baseline, not part of the original15-role pre-sampling freeze.',
        test_used_for_hyperparameter_or_checkpoint_selection=False)
    write(ROOT/'li-models-freeze.json',result);return result


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace');sys.stderr.reconfigure(encoding='utf-8',errors='backslashreplace')
    all_cases=cases()
    import preference_baseline as li
    import train_uc_baselines as nn
    config=li.PreferenceConfig()
    if (config.epochs,config.hidden_dim,config.attention_layers,config.learning_rates,config.lambda1_values,config.lambda2_values)!=(1200,128,2,(1e-4,1e-3),(.8,.9),(.1,.2)):
        raise ValueError('Native Li training grid changed')
    logs=ROOT/'fit-logs';logs.mkdir(exist_ok=True);rows=[]
    for case in all_cases:
        marker=ROOT/'models'/case.case_id/'li/complete.json'
        if marker.exists():
            _,done=nn.validate_checkpoint(ROOT,case,'li',GENERIC)
            rows.append(dict(case_id=case.case_id,status='previously_complete',fit_seconds=done['fit_seconds']));continue
        log=logs/(case.case_id+'.log')
        if log.exists():raise FileExistsError('Preserve an earlier incomplete fit attempt: '+str(log))
        started=utc();print(json.dumps(dict(event='Li_fit_started',case_id=case.case_id,utc=started,workers=1)),flush=True)
        begin=time.perf_counter()
        try:
            with log.open('x',encoding='utf-8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                nn.train_one(case,'li',ROOT,GENERIC,'cpu')
            _,done=nn.validate_checkpoint(ROOT,case,'li',GENERIC)
            row=dict(case_id=case.case_id,status='complete',started_at_utc=started,completed_at_utc=utc(),
                fit_seconds=done['fit_seconds'],fit_cpu_seconds=done['fit_cpu_seconds'],wrapper_wall_seconds=time.perf_counter()-begin,
                log=str(log),log_sha256=sha(log))
        except Exception as error:
            row=dict(case_id=case.case_id,status='failed',started_at_utc=started,completed_at_utc=utc(),
                error=repr(error),traceback=traceback.format_exc(),log=str(log),log_sha256=sha(log))
        write(log.with_suffix('.json'),row);rows.append(row);print(json.dumps(row),flush=True)
    write(logs/'complete.json',dict(results=rows,all_complete=all(row['status'] in ('complete','previously_complete') for row in rows),
        no_test_or_development_bank_opened=True))
    if any(row['status']=='failed' for row in rows):raise RuntimeError('A native Li fit failed; no setting or gate was weakened')
    frozen=freeze_models(all_cases)
    print(json.dumps(dict(event='all_three_Li_models_frozen',model_count=frozen['model_count'],sha256=sha(ROOT/'li-models-freeze.json'))),flush=True)


if __name__=='__main__':main()
