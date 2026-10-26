"""Metadata-only sequential launcher for the fixed twelve-case confirmation.

Only run after final tool-source review. All model/source/label gates are
implemented by the separately sealed evaluator and generator. No sampling,
model training, optimizer implementation or metric logic lives here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
TOOLS=ROOT.parent/'objective-gap-designs-20261025/coupled-bundle/confirmation-v1'
RUN=ROOT/'uniform-gross-confirmation'
RUNNER=TOOLS/'confirmation_runner.py'
GENERATOR=TOOLS/'fresh_bank_cli.py'
PROTOCOL=TOOLS/'fresh-confirmation-protocol.json'
GENERATOR_SHA='62a6557c74bfcc791c74bc462252f3dc8cd7ce67d4e98c2eb5e9f2f4908eab09'
PROTOCOL_SHA='adc97d04dda07e78de7cec7986deb6eb77c1bdd48560f8fcaf626ab3eb1778f0'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-runner-sha256',required=True)
    parser.add_argument('--source-capstone',type=Path,required=True)
    parser.add_argument('--audit-source',type=Path,default=ROOT/'audit_uniform_gross_confirmation.py')
    parser.add_argument('--workers',type=int,choices=(1,2,3,4),default=4)
    args=parser.parse_args()
    if sha(RUNNER)!=args.expected_runner_sha256 or sha(GENERATOR)!=GENERATOR_SHA or sha(PROTOCOL)!=PROTOCOL_SHA:
        raise ValueError('Final reviewed runner/generator/protocol differs')
    capstone=json.loads(args.source_capstone.read_text(encoding='utf-8'))
    checked=0
    for field in ('source_sha256','file_sha256','source_hashes','files'):
        for name,expected in capstone.get(field,{}).items():
            path=Path(name)
            if not path.is_absolute():path=args.source_capstone.parent/path
            if sha(path)!=expected:raise ValueError('Confirmation source capstone changed: '+str(path))
            checked+=1
    if not checked:raise ValueError('Source capstone must bind reviewed file hashes')
    logs=RUN/'execution-logs'
    logs.mkdir(parents=True,exist_ok=False)
    write(logs/'start.json',dict(status='STARTED_FIXED_TWELVE_CONFIRMATION_PIPELINE',utc=utc(),
        expected_cases=12,physical_banks=9,distinct_objectives=1080,case_queries=1440,method_returns=2880,
        workers=args.workers,threads_per_worker=1,scientific_configuration_changed=False,
        launcher_sha256=sha(__file__),source_capstone_sha256=sha(args.source_capstone),
        independent_audit_source_sha256=sha(args.audit_source) if args.audit_source.is_file() else None,
        independent_posterior_audit_required_before_handoff=True,
        runner_sha256=sha(RUNNER),generator_sha256=sha(GENERATOR),protocol_sha256=sha(PROTOCOL),
        distribution_or_seed_selection_from_confirmation=False,new_cloud_access=False))
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'

    def stage(label,argv):
        started=utc();log=logs/(label+'.log')
        with log.open('x',encoding='utf-8') as stream:
            process=subprocess.run([sys.executable,'-u',*map(str,argv)],stdout=stream,stderr=subprocess.STDOUT,env=env)
        receipt=dict(stage=label,argv=list(map(str,argv)),started_at_utc=started,completed_at_utc=utc(),
            exit_code=process.returncode,log=str(log),log_sha256=sha(log))
        write(log.with_suffix('.json'),receipt);print(json.dumps(receipt),flush=True)
        if process.returncode:raise RuntimeError('Pipeline stops at failed stage: '+label)

    freeze=RUN/'root_all_models_freeze.json'
    stage('01-wrapper-self-test',[RUNNER,'self-test','--output',logs/'wrapper-self-test-output.json'])
    frozen_extras=[GENERATOR,TOOLS/'confirmation-wrapper-self-test-final.json',args.source_capstone,Path(__file__)]
    if args.audit_source.is_file():frozen_extras.append(args.audit_source)
    stage('02-all-model-freeze',[RUNNER,'freeze','--freeze',freeze,'--include-source',*frozen_extras,
        '--output',logs/'freeze-command-output.json'])
    expected=sha(freeze)
    write(logs/'expected-freeze-sha256.json',dict(model_freeze_sha256=expected,utc=utc()))
    stage('03-all-model-validation',[RUNNER,'validate','--freeze',freeze,'--output',logs/'validation-command-output.json'])
    stage('04-generator-preflight',[GENERATOR,'--freeze',freeze,'--expected-freeze-sha256',expected,'--validate-only'])
    stage('05-generate-all-physical-banks',[GENERATOR,'--freeze',freeze,'--expected-freeze-sha256',expected,'--generate'])
    stage('06-evaluate-all-twelve-cases',[RUNNER,'evaluate','--freeze',freeze,'--workers',args.workers,
        '--output',RUN/'evaluation-summary.json'])
    write(logs/'complete.json',dict(status='ALL_FIXED_CONFIRMATION_RETURNS_WRITTEN_AWAITING_INDEPENDENT_AUDIT',
        utc=utc(),model_freeze_sha256=expected,case_count=12,physical_bank_count=9,
        distinct_objectives=1080,case_queries=1440,method_return_count=2880))


if __name__=='__main__':main()
