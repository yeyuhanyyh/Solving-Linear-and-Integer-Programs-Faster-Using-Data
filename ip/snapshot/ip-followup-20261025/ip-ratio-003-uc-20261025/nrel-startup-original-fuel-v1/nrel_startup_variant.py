"""Original-fuel synthetic UC with capacity-scaled startup-charge robustness.

NREL2012 Table1-2 motivates only a $/MW order of magnitude for a capital and
maintenance lower-bound component. The constructed Uniform range is not an
empirical distribution, a full startup-fuel estimate or calibrated plant data.
Original UC generator, solver and SVM/MLOPT sources remain unchanged.
"""
from __future__ import annotations
import os
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[key]="1"
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PRODUCER=ROOT/"uc_reference_screen.py"
HELPER=ROOT/"baseline_helpers.py"
spec=importlib.util.spec_from_file_location("readonly_uc_ratio_screen",PRODUCER)
screen=importlib.util.module_from_spec(spec);sys.modules[spec.name]=screen;spec.loader.exec_module(screen)
SEEDS=[101,211,307]
SETTING=dict(name="G4T12_all_units_nrel_startup",G=4,T=12,scope="all_units")


def new_physical(setting,seed):
    G,T=setting["G"],setting["T"];q=G*T
    if (G,T,setting["scope"])!=(4,12,"all_units"):raise ValueError("Fixed three-seed G4 all-unit robustness only")
    base=screen.uc.make_uc_base(seed,n_units=G,n_times=T)
    old_startup=base["metadata"]["startup_cost"].copy()
    rate=22.+(old_startup-220.)*38./630.
    startup=base["metadata"]["capacity"]*rate
    base["metadata"]["startup_cost"]=startup
    base["base_cost"][q:2*q]=np.tile(startup,T)
    original_fraction=base["metadata"]["demand"]/sum(base["metadata"]["capacity"])
    fraction=.30+.20*(original_fraction-original_fraction.min())/(original_fraction.max()-original_fraction.min())
    demand=fraction*sum(base["metadata"]["capacity"])
    for t,value in enumerate(demand):
        row=base["metadata"]["row_names"].index(f"balance[{t}]")
        base["row_lower"][row]=base["row_upper"][row]=value
    base["metadata"].update(demand=demand,demand_fraction=fraction,
        demand_mapping="Original smooth profile mapped to total-capacity fractions[0.30,0.50]",
        original_startup_draw=old_startup,startup_charge_rate_per_MW=rate,
        startup_charge_mapping="capacity_g*(22+(oldstartup_g-220)*38/630); synthetic Uniform[22,60] rate",
        fuel_cost_unchanged_from_original_22_to_45=True,
        cost_units_interpretation="If p is MW and one period one hour, linear dispatch coefficients are currency/MWh, on cost currency/hour and startup cost currency/start",
        startup_reference="NREL2012 Power Plant Cycling Costs Table1-2: capital-and-maintenance lower-bound component order of magnitude only",
        startup_reference_url="https://docs.nrel.gov/docs/fy12osti/55433.pdf",
        not_a_calibrated_plant_or_full_startup_fuel_model=True,
        objective_note="Original total dispatch/no-load/startup/shutdown objective retained; no constant subtracted")
    assert np.all((base["metadata"]["fuel_cost"]>=22)&(base["metadata"]["fuel_cost"]<=45))
    assert np.all((rate>=22)&(rate<=60))
    if not screen.uc.validate_uc_solution(base,screen.uc.all_on_feasible_solution(base))["valid"]:
        raise ValueError("Original-fuel robustness lost all-on physical feasibility")
    offset=np.zeros(4*q);coeff=np.zeros((4*q,3*G))
    for component in range(4):
        for t in range(T):
            for g in range(G):coeff[component*q+t*G+g,3*g+t//4]=base["base_cost"][component*q+t*G+g]
    return base,offset,coeff,list(range(G))


def main():
    out=HERE/"run-v1"
    if out.exists():raise FileExistsError("Immutable original-fuel startup robustness already exists")
    out.mkdir()
    protocol=dict(created_utc=screen.now(),setting=SETTING,seeds=SEEDS,training_prefixes=[20,32],
        exploratory_training_label_count=64,rank_diagnostic_prefixes=[20,32,40,64],development_count=24,
        physical_cost_change="Keep fuelUniform[22,45], onUniform[80,240], shutdownUniform[5,30]. Set startup=capacity_g*rate_g with rate_g Uniform[22,60] by affine mapping of the original uniform startup draw",
        reference_title="Power Plant Cycling Costs, NREL2012, Table1-2",reference_url="https://docs.nrel.gov/docs/fy12osti/55433.pdf",
        citation_boundary="Reported numbers describe a capital-and-maintenance lower-bound component, not complete startup fuel. The Uniform[22,60] synthetic range is inspired by the $/MW scale; no calibrated-plant or empirical-distribution claim",
        law="Independent LogUniform[1/3,3] multipliers, every generator x three four-period blocks, shared by all four cost components",
        paired_multipliers="Same predeclared training/development feature streams as previous G4all screen by namespace alias only; neither old bank nor old returned solutions read",
        all_seeds_retained=True,no_additional_setting_search=True,global_optimizer_hull_K_not_proven=True,
        objective="Full original optimal totalcost/returned totalcost, no subtraction or arbitrary shift",
        source_sha256={str(p.resolve()):screen.sha(p) for p in [Path(__file__),PRODUCER,HELPER,screen.CODE/"controls"/"uc_model.py",screen.CODE/"controls"/"experiment_core.py",screen.CODE/"portable_adapter.py",screen.CODE/"mlopt_baseline.py"]},
        no_NeuralDiving_or_Li_training=True,timing_not_isolated=True,no_confirmation_claim=True)
    screen.write_new(out/"protocol.json",protocol)
    spec=importlib.util.spec_from_file_location("uc_nrel_baseline_helpers",HELPER)
    helpers=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helpers;spec.loader.exec_module(helpers)
    helpers._runtime()
    original_rng=screen.rng
    screen.physical=new_physical
    screen.rng=lambda label:original_rng(label.replace("all_units_nrel_startup","all_units"))
    rows=[]
    for seed in SEEDS:
        try:
            new=screen.run_setting(out,SETTING,seed,helpers);rows.extend(new)
            print(json.dumps(dict(event="nrel_startup_robustness_case_complete",seed=seed,
                means=[dict(n=r["n_train"],method=r["method"],K=r["K"],ratio=r["objective_ratio_mean"],valid=r["valid_count"]) for r in new])),flush=True)
        except Exception as error:
            screen.write_new(out/f"seed{seed}-failure.json",dict(seed=seed,error=repr(error),retained=True,no_seed_or_cost_replacement=True))
            print(json.dumps(dict(event="nrel_startup_robustness_failure",seed=seed,error=repr(error))),flush=True)
    screen.write_new(out/"summary.json",dict(status="COMPLETE_WITH_RETAINED_FAILURES" if list(out.glob("*-failure.json")) else "COMPLETE",
        rows=rows,created_utc=screen.now(),protocol_sha256=screen.sha(out/"protocol.json"),
        source_sha256={str(p.resolve()):screen.sha(p) for p in out.rglob("*") if p.is_file()},
        exploratory_only=True,no_confirmation_claim=True))
    print(json.dumps(dict(status="NREL_STARTUP_ROBUSTNESS_COMPLETE",rows=len(rows),summary_sha256=screen.sha(out/"summary.json"))))


if __name__=="__main__":main()
