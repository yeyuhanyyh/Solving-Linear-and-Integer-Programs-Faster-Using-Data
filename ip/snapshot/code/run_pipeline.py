"""Remote stage launcher with a bounded wall-clock watchdog.

This stops experiment child processes at the local deadline. Provider billing
must additionally be stopped by deleting the rented instance through the
user-controlled provider account; this package contains no provider token.
"""
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=True),encoding="utf-8")


def terminate_process_group(process, grace_seconds=20.):
    """Stop the recorded process group even if its leader has already exited."""
    if os.name!="posix":
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=max(0.,grace_seconds))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return
    pgid=process.pid  # start_new_session=True makes the child the group leader.
    def group_exists():
        try:
            os.killpg(pgid,0)
            return True
        except ProcessLookupError:
            return False
    try:
        os.killpg(pgid,signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    until=time.monotonic()+max(0.,grace_seconds)
    while group_exists() and time.monotonic()<until:
        process.poll()  # Reap the leader without treating that as group exit.
        time.sleep(.05)
    if group_exists():
        try:
            os.killpg(pgid,signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly-price-usd",type=float,required=True,help="Actual verified rental price, including provider charges")
    parser.add_argument("--max-hours",type=float)
    parser.add_argument("--deadline-utc",help="Provider deadline ISO timestamp; setup/upload time then also counts")
    parser.add_argument("--stage",choices=("all","train","benchmark","summary","validate","smoke"),default="all")
    parser.add_argument("--cases",nargs="+")
    parser.add_argument("--device",choices=("cuda","cpu","auto"),default="cuda")
    parser.add_argument("--allow-cpu-training",action="store_true",help="Explicitly permit substantial CPU fits when CUDA is unavailable")
    args=parser.parse_args()
    protocol=json.loads((ROOT/"protocol.json").read_text(encoding="utf-8"))
    resource=protocol["resource"]
    hourly=args.hourly_price_usd
    budget=float(resource["budget_usd"])
    reserve=float(resource.get("reserve_usd",5.))
    if not 0<hourly<=float(resource["maximum_hourly_price_usd"]):
        raise ValueError("Verified hourly price is outside the authorized cap")
    hours=min(float(resource["max_wall_hours"]),(budget-reserve)/hourly)
    if args.max_hours is not None:
        if not 0<args.max_hours<=hours:
            raise ValueError("Requested wall time exceeds the authorized budget-derived cap")
        hours=args.max_hours
    now=time.time()
    deadline=now+hours*3600
    if args.deadline_utc:
        specified=datetime.fromisoformat(args.deadline_utc.replace("Z","+00:00"))
        if specified.tzinfo is None:
            raise ValueError("Provider deadline must include UTC timezone")
        deadline=min(deadline,specified.timestamp())
    if deadline<=now:
        raise ValueError("Provider deadline has already passed")
    write(ROOT/"logs"/"budget-watchdog.json",dict(hourly_price_usd=hourly,budget_usd=budget,reserve_usd=reserve,
        maximum_compute_hours=hours,started_utc=datetime.now(timezone.utc).isoformat(),
        deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
        provider_instance_deletion_required=True,provider_credentials_present_in_package=False))
    commands=[]
    case_args=[] if args.cases is None else ["--cases",*args.cases]
    if args.stage in ("all","validate"):
        commands.append([sys.executable,"code/run_training.py","--validate-only","--device",args.device,*case_args])
    if args.stage in ("all","smoke"):
        commands.append([sys.executable,"code/gpu_smoke.py","--device",args.device])
    if args.stage in ("all","train"):
        cpu_args=["--allow-cpu-training"] if args.allow_cpu_training else []
        commands.append([sys.executable,"code/run_training.py","--device",args.device,*cpu_args,*case_args])
    if args.stage in ("all","benchmark"):
        commands.append([sys.executable,"code/run_benchmark.py",*case_args])
    if args.stage in ("all","summary"):
        commands.append([sys.executable,"code/summarize_results.py",*case_args])
    child=None
    interrupted=False
    def stop(signum=None,frame=None):
        nonlocal interrupted
        interrupted=True
        if child is not None:
            if os.name=="posix":
                try:
                    os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif child.poll() is None:
                child.terminate()
    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum,stop)
    try:
        for index,command in enumerate(commands):
            if interrupted or time.time()>=deadline:
                raise TimeoutError("Experiment deadline reached before next phase")
            log=ROOT/"logs"/f"pipeline-{int(now)}-phase{index}.log"
            with log.open("w",encoding="utf-8") as output:
                print(json.dumps(dict(event="phase_start",command=command,log=log.relative_to(ROOT).as_posix())),flush=True)
                child=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
                    start_new_session=(os.name=="posix"))
                while child.poll() is None:
                    if interrupted or time.time()>=deadline:
                        stop()
                        terminate_process_group(child,grace_seconds=min(20.,max(0.,deadline-time.time())))
                        raise TimeoutError("Experiment stopped at its authorized wall-clock deadline; partial artifacts retained")
                    time.sleep(1.)
                if interrupted or time.time()>=deadline:
                    raise TimeoutError("Experiment interrupted or reached its deadline as a phase exited")
                if child.returncode:
                    raise RuntimeError(f"Phase failed with code {child.returncode}; inspect {log.relative_to(ROOT)}")
                print(json.dumps(dict(event="phase_complete",command=command)),flush=True)
                child=None
        write(ROOT/"logs"/"pipeline-complete.json",dict(completed_utc=datetime.now(timezone.utc).isoformat(),
            phases=len(commands),elapsed_seconds=time.time()-now,stop_and_delete_provider_instance_now=True))
    except BaseException as exc:
        stop()
        if child is not None:
            terminate_process_group(child,grace_seconds=min(20.,max(0.,deadline-time.time())))
        write(ROOT/"logs"/"pipeline-blocked.json",dict(error=str(exc),elapsed_seconds=time.time()-now,
            deadline_reached=time.time()>=deadline,partial_artifacts_retained=True,
            stop_and_delete_provider_instance_now=True))
        raise


if __name__=="__main__":
    main()
