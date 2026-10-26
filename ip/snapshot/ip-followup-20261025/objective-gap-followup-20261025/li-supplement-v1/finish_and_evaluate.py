"""Metadata-only completion after serial/parallel native fits have both ended."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    spec=importlib.util.spec_from_file_location('completed_original_li_training',ROOT/'train_li_supplement.py')
    trainer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=trainer;spec.loader.exec_module(trainer)
    cases=trainer.cases()
    import train_uc_baselines as nn
    rows=[]
    for case in cases:
        _,marker=nn.validate_checkpoint(ROOT,case,'li',trainer.GENERIC)
        if marker['sanity_gate_passed'] is not True or marker['training_label_count']!=36:
            raise ValueError('A native Li checkpoint is incomplete')
        rows.append(dict(case_id=case.case_id,seed=case.seed,fit_seconds=marker['fit_seconds'],
            fit_cpu_seconds=marker['fit_cpu_seconds'],checkpoint_sha256=trainer.sha(ROOT/'models'/case.case_id/'li/complete.json')))
    frozen=ROOT/'li-models-freeze.json'
    if not frozen.exists():trainer.freeze_models(cases)
    else:
        value=trainer.read(frozen)
        for path,expected in value['source_sha256'].items():
            if trainer.sha(path)!=expected:raise ValueError('Frozen Li source changed')
        for entry in value['cases'].values():
            for path,expected in entry['files'].items():
                if trainer.sha(path)!=expected:raise ValueError('Frozen Li artifact changed')
    trainer.write(ROOT/'combined-scheduler-complete.json',dict(status='ALL_THREE_NATIVE_LI_FITS_COMPLETE',
        models=rows,model_freeze_sha256=trainer.sha(frozen),
        schedule='Original serial worker1009→1013; separately authorized worker1019. The exclusive1019 log prevents a duplicate serial fit at its safe pre-fit boundary.',
        original_protocol_sha256=trainer.sha(trainer.PROTOCOL),maximum_knapsack_workers_override=2,
        scientific_training_source_unchanged=True,training_source_sha256=trainer.sha(ROOT/'train_li_supplement.py')))
    log=ROOT/'evaluation.log'
    with log.open('x',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-u',str(ROOT/'evaluate_li_supplement.py')],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode==0:
        quality=trainer.read(ROOT/'results/pooled-quality.json')
        print(json.dumps(dict(status=quality['status'],cells=quality['cells'],per_seed=quality['per_seed'],
            report=str(ROOT/'results/pooled-quality.json'),sha256=trainer.sha(ROOT/'results/pooled-quality.json'))),flush=True)
    else:print(str(log),flush=True)
    if result.returncode:raise RuntimeError('Li evaluation failed; see preserved evaluation.log')


if __name__=='__main__':main()
