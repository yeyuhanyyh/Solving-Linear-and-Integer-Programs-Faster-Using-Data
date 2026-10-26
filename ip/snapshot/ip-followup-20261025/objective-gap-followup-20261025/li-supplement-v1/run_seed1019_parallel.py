"""Scheduler-only one-case handoff; same immutable trainer/native Li fit."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace');sys.stderr.reconfigure(encoding='utf-8',errors='backslashreplace')
    source=ROOT/'train_li_supplement.py'
    spec=importlib.util.spec_from_file_location('unchanged_li_training_scheduler_handoff',source)
    trainer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=trainer;spec.loader.exec_module(trainer)
    cases=trainer.cases();case=next(case for case in cases if case.seed==1019)
    import train_uc_baselines as nn
    marker=ROOT/'models'/case.case_id/'li/complete.json'
    if marker.exists():
        nn.validate_checkpoint(ROOT,case,'li',trainer.GENERIC)
        print(json.dumps(dict(status='ALREADY_COMPLETE_NO_FIT',case_id=case.case_id)),flush=True);return
    log=ROOT/'fit-logs'/(case.case_id+'.log');started=trainer.utc()
    # This exclusive file is also the existing serial scheduler's safe
    # pre-fit boundary. If it arrives before this fit is done, it refuses a
    # duplicate before invoking train_one. No scientific source is modified.
    with log.open('x',encoding='utf-8') as stream:
        trainer.write(ROOT/'parallel-scheduler-handoff.json',dict(case_id=case.case_id,seed=1019,
            started_at_utc=started,workers_in_this_process=1,maximum_knapsack_workers=2,
            immutable_trainer_sha256=trainer.sha(source),scheduler_sha256=trainer.sha(__file__),
            safe_duplicate_barrier=str(log),scientific_settings_changed=False))
        print(json.dumps(dict(event='parallel_Li1019_fit_started',case_id=case.case_id,utc=started)),flush=True)
        begin=time.perf_counter()
        try:
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                nn.train_one(case,'li',ROOT,trainer.GENERIC,'cpu')
            _,marker=nn.validate_checkpoint(ROOT,case,'li',trainer.GENERIC)
            result=dict(case_id=case.case_id,status='complete',started_at_utc=started,completed_at_utc=trainer.utc(),
                fit_seconds=marker['fit_seconds'],fit_cpu_seconds=marker['fit_cpu_seconds'],
                wrapper_wall_seconds=time.perf_counter()-begin,log=str(log))
        except Exception as error:
            result=dict(case_id=case.case_id,status='failed',error=repr(error),traceback=traceback.format_exc(),
                started_at_utc=started,completed_at_utc=trainer.utc(),log=str(log))
    result['log_sha256']=trainer.sha(log)
    trainer.write(log.with_suffix('.json'),result);print(json.dumps(result),flush=True)
    if result['status']!='complete':raise RuntimeError('Native Li1019 failed; no gate/settings changed')


if __name__=='__main__':main()
