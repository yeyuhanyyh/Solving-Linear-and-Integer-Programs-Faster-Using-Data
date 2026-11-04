"""Outer-return budget checks for all vendored ND validation/QA calls."""
import time
import neural_diving_baseline as nd

_source_solve=nd.solve_cost


def checked_return(record,elapsed,budget):
    record=dict(record)
    record["source_seconds"]=record.get("seconds")
    record["seconds"]=float(elapsed)
    record["within_return_budget"]=elapsed<=budget
    record["raw_valid_before_return_budget"]=bool(record.get("valid"))
    record["valid"]=bool(record.get("valid") and record["within_return_budget"])
    if not record["within_return_budget"]:
        record["status_before_return_budget"]=record.get("status")
        record["status"]="return_budget_exceeded"
    return record


def checked_solve(model,case,cost,seed,total_time_limit=30.,solve_fn=None,mode="native",K=None):
    started=time.perf_counter()
    record=_source_solve(model,case,cost,seed,total_time_limit,solve_fn,mode,K)
    return checked_return(record,time.perf_counter()-started,total_time_limit)


def install():
    nd.solve_cost=checked_solve
