"""ND-only root-LP feature subset, using an unrestricted continuous HiGHS LP.

No cached integer optimizer, labels, test reference, or variable IDs enter these
features. A fractional primal solution is valid for this relaxation. Certified
optimum, primal/dual validity and original continuous KKT checks are required.
"""
from __future__ import annotations
import time
from typing import Any
import numpy as np
from scipy import sparse
import highspy

ROOT_FEATURE_NAMES = (
    "lp_solution_scaled", "lp_fractionality", "lp_reduced_cost_normalized",
    "lp_at_lower_bound", "lp_at_upper_bound", "lp_basis_lower", "lp_basis_basic",
    "lp_basis_upper", "lp_basis_zero", "lp_basis_nonbasic",
    "lp_row_dual_normalized", "lp_row_slack_scaled", "lp_row_at_lower",
    "lp_row_at_upper", "lp_row_objective_cosine")
FEATURE_SOURCE = "https://raw.githubusercontent.com/ds4dm/learn2branch/master/utilities.py"


class RootLPFeatureError(RuntimeError):
    def __init__(self,message: str,metadata: dict[str,Any]):
        super().__init__(message)
        self.metadata=metadata


class RootLPFeatureExtractor:
    def __init__(self,model: Any,seed: int=1909,time_limit: float=5.):
        raw=getattr(model,"Acsr",None)
        if raw is None:
            raw=getattr(model,"A",None)
        self.A=sparse.csr_matrix(raw,dtype=float)
        self.row_lower=np.asarray(model.row_lower,dtype=float)
        self.row_upper=np.asarray(model.row_upper,dtype=float)
        self.col_lower=np.asarray(model.col_lower,dtype=float)
        self.col_upper=np.asarray(model.col_upper,dtype=float)
        self.integrality=np.asarray(model.integrality)!=0
        self.seed,self.time_limit=int(seed),float(time_limit)
        self.n_rows,self.n_columns=self.A.shape
        self.row_norm=np.sqrt(np.asarray(
            self.A.multiply(self.A).sum(axis=1)).ravel())
        # Gasse's feature convention uses unit norm for an exactly zero row.
        # A tiny epsilon would turn an ordinary zero-row slack into a huge input.
        self.row_norm[self.row_norm==0.]=1.
        finite_lb=np.where(np.isfinite(self.col_lower),np.abs(self.col_lower),0.)
        finite_ub=np.where(np.isfinite(self.col_upper),np.abs(self.col_upper),0.)
        self.col_scale=np.maximum(np.maximum(finite_lb,finite_ub),1.)
        if self.time_limit<=0:
            raise ValueError("root-LP time limit must be positive")

    def extract(self,cost: np.ndarray,time_limit: float|None=None
                )->tuple[np.ndarray,dict[str,Any]]:
        started=time.perf_counter()
        c=np.asarray(cost,dtype=float)
        if c.shape!=(self.n_columns,) or not np.isfinite(c).all():
            raise ValueError("root-LP cost has invalid shape/entries")
        cap=self.time_limit if time_limit is None else min(self.time_limit,float(time_limit))
        if cap<=0:
            raise RootLPFeatureError("root-LP inference has no remaining budget",
                                     {"status":"budget_exhausted","seconds":0.})
        lp=highspy.HighsLp()
        lp.num_col_,lp.num_row_=self.n_columns,self.n_rows
        lp.col_cost_,lp.col_lower_,lp.col_upper_=c.copy(),self.col_lower.copy(),self.col_upper.copy()
        lp.row_lower_,lp.row_upper_=self.row_lower.copy(),self.row_upper.copy()
        # Omit all original integer domains: this is the unrestricted root LP.
        lp.integrality_=[highspy.HighsVarType.kContinuous]*self.n_columns
        lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_=self.A.indptr.astype(np.int32)
        lp.a_matrix_.index_=self.A.indices.astype(np.int32)
        lp.a_matrix_.value_=self.A.data
        attempts=[]
        def metadata(record):
            result=dict(record)
            result.update(attempts=[dict(a) for a in attempts],
                seconds=time.perf_counter()-started,
                build_seconds=sum(a.get("build_seconds",0.) for a in attempts),
                solve_seconds=sum(a.get("solve_seconds",0.) for a in attempts),
                time_limit_seconds=cap,solver_seed=self.seed,source=FEATURE_SOURCE,
                certificate_scope="continuous_root_relaxation",
                budget_scope="Combined model construction, both attempts, certification and feature assembly",
                presolve_off_retry_used=len(attempts)>1)
            return result
        def fail(message,record):
            raise RootLPFeatureError(message,metadata(record))
        # Keep the original solve first. A single same-LP retry is permitted
        # only for an otherwise valid certificate with unsupported dual signs.
        # Neither its objective nor any multiplier is projected or adjusted.
        for attempt_index in range(2):
            attempt_started=started if attempt_index==0 else time.perf_counter()
            record={"attempt":attempt_index+1,"presolve":"default" if attempt_index==0 else "off",
                    "status":"not_run","certified":False,"solve_seconds":0.}
            attempts.append(record)
            if cap-(time.perf_counter()-started)<=0:
                record["status"]="budget_exhausted_before_build"
                fail("root-LP combined budget exhausted before an attempt",record)
            h=highspy.Highs()
            options=dict(output_flag=False,threads=1,random_seed=self.seed,
                solver="simplex",primal_feasibility_tolerance=1e-7,dual_feasibility_tolerance=1e-7)
            if attempt_index:
                options["presolve"]="off"
            for key,value in options.items():
                if h.setOptionValue(key,value)!=highspy.HighsStatus.kOk:
                    raise RuntimeError(f"root-LP HiGHS option rejected:{key}")
            if h.passModel(lp)!=highspy.HighsStatus.kOk:
                raise RuntimeError("root-LP HiGHS model rejected")
            record["build_seconds"]=time.perf_counter()-attempt_started
            remaining=cap-(time.perf_counter()-started)
            if remaining<=0:
                record["status"]="budget_exhausted_before_run"
                fail("root-LP combined budget exhausted before a solve",record)
            if h.setOptionValue("time_limit",remaining)!=highspy.HighsStatus.kOk:
                raise RuntimeError("root-LP HiGHS option rejected:time_limit")
            # Option setting and solver preparation also consume the same cap.
            if cap-(time.perf_counter()-started)<=0:
                record["status"]="budget_exhausted_before_run"
                fail("root-LP combined budget exhausted before a solve",record)
            record["solver_time_limit_seconds"]=remaining
            solve_started=time.perf_counter()
            run=h.run()
            record["solve_seconds"]=time.perf_counter()-solve_started
            status=h.getModelStatus()
            sol,info,basis=h.getSolution(),h.getInfo(),h.getBasis()
            record.update(status=h.modelStatusToString(status),
                optimal=status==highspy.HighsModelStatus.kOptimal,run_status=str(run),
                value_valid=bool(sol.value_valid),dual_valid=bool(sol.dual_valid),basis_valid=bool(basis.valid))
            if cap-(time.perf_counter()-started)<=0:
                record["budget_exhausted"]=True
                fail("root-LP combined budget exhausted during a solve",record)
            if run==highspy.HighsStatus.kError or not (
                    record["optimal"] and sol.value_valid and sol.dual_valid and basis.valid):
                fail("root LP did not provide an optimal primal/dual/basis",record)
            certificate_started=time.perf_counter()
            x=np.asarray(sol.col_value,dtype=float)
            y=np.asarray(sol.row_dual,dtype=float)
            rc=np.asarray(sol.col_dual,dtype=float)
            ax=self.A@x
            if not all(np.isfinite(v).all() for v in (x,y,rc,ax)):
                fail("root LP returned nonfinite primal/dual data",record)
            violation=max(float(np.max(self.row_lower-ax,initial=0.)),
                          float(np.max(ax-self.row_upper,initial=0.)),
                          float(np.max(self.col_lower-x,initial=0.)),
                          float(np.max(x-self.col_upper,initial=0.)))
            stationarity=float(np.max(np.abs(c-self.A.T@y-rc),initial=0.))
            scale=max(1.,float(np.max(np.abs(c),initial=0.)))
            dual_tol=1e-7*scale
            signs_ok=True
            dual_objective=0.
            unsupported=[]
            for kind,values,lower,upper in (("row",y,self.row_lower,self.row_upper),
                                           ("column",rc,self.col_lower,self.col_upper)):
                positive=values>0;negative=values<0
                # Even a tiny signed multiplier requires a finite endpoint.
                bad=(positive&~np.isfinite(lower))|(negative&~np.isfinite(upper))
                if bad.any():
                    signs_ok=False
                    unsupported.extend({"kind":kind,"index":int(i),"value":float(values[i]),
                        "missing_endpoint":"lower" if positive[i] else "upper"}
                        for i in np.flatnonzero(bad))
                good_pos=positive&np.isfinite(lower)
                good_neg=negative&np.isfinite(upper)
                dual_objective+=float(values[good_pos]@lower[good_pos]+values[good_neg]@upper[good_neg])
            objective=float(c@x)
            objective_gap=abs(objective-dual_objective)
            objective_tolerance=1e-6+1e-9*abs(objective)
            objective_consistency=abs(objective-float(info.objective_function_value))
            other_checks=bool(violation<=1e-6 and stationarity<=dual_tol and
                objective_gap<=objective_tolerance and objective_consistency<=objective_tolerance)
            certified=bool(other_checks and signs_ok)
            record.update({"continuous_feasibility_violation":violation,
                "integer_residual_diagnostic":float(np.max(np.abs(x[self.integrality]-
                    np.rint(x[self.integrality])),initial=0.)),
                "dual_stationarity_residual":stationarity,"dual_stationarity_tolerance":dual_tol,
                "dual_signs_valid":signs_ok,"unsupported_dual_support":unsupported,
                "primal_objective":objective,"dual_objective":dual_objective,"primal_dual_gap":objective_gap,
                "objective_tolerance":objective_tolerance,"objective_consistency_residual":objective_consistency,
                "certificate_except_dual_signs_valid":other_checks,"certified":certified,
                "certificate_seconds":time.perf_counter()-certificate_started})
            if cap-(time.perf_counter()-started)<=0:
                record["budget_exhausted"]=True
                fail("root-LP combined budget exhausted during certification",record)
            if certified:
                meta=metadata(record)
                meta["accepted_attempt"]=attempt_index+1
                break
            if attempt_index==0 and other_checks and not signs_ok:
                record["retry_trigger"]="unsupported_dual_signs_only"
                continue
            fail("root-LP continuous KKT certificate failed",record)
        feature_started=time.perf_counter()
        n,m=self.n_columns,self.n_rows
        features=np.zeros((n+m,len(ROOT_FEATURE_NAMES)),dtype=np.float64)
        objective_norm=float(np.linalg.norm(c))
        if objective_norm==0.:
            objective_norm=1.
        features[:n,0]=x/self.col_scale
        features[:n,1]=np.where(self.integrality,np.abs(x-np.rint(x)),0.)
        features[:n,2]=rc/objective_norm
        features[:n,3]=np.isfinite(self.col_lower)&(np.abs(x-self.col_lower)<=1e-6)
        features[:n,4]=np.isfinite(self.col_upper)&(np.abs(x-self.col_upper)<=1e-6)
        basis_codes={highspy.HighsBasisStatus.kLower:5,highspy.HighsBasisStatus.kBasic:6,
                     highspy.HighsBasisStatus.kUpper:7,highspy.HighsBasisStatus.kZero:8,
                     highspy.HighsBasisStatus.kNonbasic:9}
        for j,value in enumerate(basis.col_status):
            features[j,basis_codes[value]]=1.
        features[n:,10]=y/(self.row_norm*objective_norm)
        lower_slack=np.where(np.isfinite(self.row_lower),ax-self.row_lower,np.inf)
        upper_slack=np.where(np.isfinite(self.row_upper),self.row_upper-ax,np.inf)
        slack=np.minimum(lower_slack,upper_slack)
        features[n:,11]=np.where(np.isfinite(slack),np.maximum(slack,0.)/self.row_norm,0.)
        features[n:,12]=np.isfinite(self.row_lower)&(np.abs(ax-self.row_lower)<=1e-6)
        features[n:,13]=np.isfinite(self.row_upper)&(np.abs(ax-self.row_upper)<=1e-6)
        features[n:,14]=(self.A@c)/(self.row_norm*objective_norm)
        if not np.isfinite(features).all():
            raise RootLPFeatureError("root-LP feature assembly returned nonfinite values",meta)
        with np.errstate(over="ignore",invalid="ignore"):
            output=features.astype(np.float32)
        if not np.isfinite(output).all():
            meta["seconds"]=time.perf_counter()-started
            raise RootLPFeatureError("root-LP features exceed finite float32 range",meta)
        meta["feature_assembly_seconds"]=time.perf_counter()-feature_started
        meta["seconds"]=time.perf_counter()-started
        if meta["seconds"]>cap:
            meta["budget_exhausted"]=True
            raise RootLPFeatureError("root-LP combined budget exhausted during feature assembly",meta)
        meta["feature_names"]=list(ROOT_FEATURE_NAMES)
        return output,meta

