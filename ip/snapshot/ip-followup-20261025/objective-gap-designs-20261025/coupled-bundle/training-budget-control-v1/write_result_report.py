"""Write a compact report from existing checked results; no training or solves."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
FOLLOWUP = ROOT.parents[2]
RUN = FOLLOWUP / 'objective-gap-followup-20261025/uniform-gross-stream2-training-budget-control'
AUDIT = RUN / 'independent-lightweight-check.json'
ORIGINAL = FOLLOWUP / 'objective-gap-followup-20261025/uniform-gross-g32-n36-pilot-results'
EXTRAS = FOLLOWUP / 'objective-gap-followup-20261025/extra-baselines-coupled-v1'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def link(label,p): return f'[{label}](<{p.resolve().as_posix()}>)'

audit = read(AUDIT)
assert audit['status'] == 'COMPLETE_CHECKED' and audit['discrepancy_count'] == 0
assert audit['reference_count'] == 360 and audit['raw_return_count'] == 1800
train = []
for seed in (1009,1013,1019):
    cid = f'coupledbundle_uniformgross_random32_b0.5_seed{seed}_n36'
    op = ORIGINAL / 'models' / cid / 'neural_diving/metadata.json'
    cp = ROOT / 'continued-models' / cid / 'metadata.json'
    original, continued = read(op), read(cp)
    inner = original['metadata']
    train.append(dict(seed=seed,case_id=cid,hidden_size=inner['selected_width'],parameter_count=inner['parameter_count'],
        original_fit_wall_seconds=original['fit_seconds'],original_fit_CPU_seconds=original['fit_cpu_seconds'],
        original_final_refit_seconds=inner['final_refit_seconds'],additional_optimizer_steps=2000,
        additional_wall_seconds=continued['total_wall_seconds'],additional_CPU_seconds=continued['total_CPU_seconds'],
        combined_recorded_wall_seconds=original['fit_seconds']+continued['total_wall_seconds'],
        initial_train_bit_error=continued['trajectory'][0]['diagnostics']['expected_map_bit_error'],
        final_train_bit_error=continued['trajectory'][-1]['diagnostics']['expected_map_bit_error'],
        original_metadata_path=str(op.resolve()),original_metadata_sha256=sha(op),
        continued_metadata_path=str(cp.resolve()),continued_metadata_sha256=sha(cp)))

extra_times = [x for x in read(EXTRAS/'learning-times.json')['roles'] if '_random32_' in x['case_id']]
pooled = dict(schema='stream2-fixed-training-budget-result-summary-v1',
    setting=dict(problem='coupled-bundle project selection with local knapsack capacities and global upgrade conflicts',
        G=32,D=96,K=32,n_train=36,seeds=[1009,1013,1019],test_costs_per_seed=120,distinct_test_costs=360,
        upgrade_gross_return_law='Independent Uniform[0.1,2.5]',base_project_gross_return_each=0.25,
        prior_exploratory_labels_collected_per_seed=69,labels_used_in_each_fit=36),
    pooled=audit['cells'],per_seed=audit['per_seed'],paired_valid_ND_diagnostic=audit['paired_valid_ND_diagnostic'],
    neural_diving_training=train,svm_mlopt_learning_times=extra_times,
    interpretation=dict(ratio_denominator='Valid, timely returned original feasible solutions; N is disclosed separately.',
        recovery_denominator='All 360 test costs for each method.',
        canonical='Diagnostic original-feasible completion keeping the returned C decisions, not a returned solution; may violate predicted fixings.',
        training_control='Fixed final checkpoint after exactly 2000 extra updates, not best-validation or best-test selection.',
        limitation='This control does not prove global neural-network convergence or a generalization bound.',
        timing='Recorded parallel-run times exclude training-label generation and are not isolated speedup benchmarks.'),
    receipts={str(p.resolve()):sha(p) for p in [AUDIT,RUN/'all-five-role-models-freeze.json',ROOT/'control-source-seal.json',ROOT/'continuation-training-summary.json',EXTRAS/'g32-model-freeze.json',EXTRAS/'learning-times.json']})

table = ['| Method | Recovery (%) | Objective ratio | Ratio N / 360 |', '|---|---:|---:|---:|']
for x in audit['cells']:
    table.append(f"| {x['method']} | {x['recovery_percent']:.2f} | {x['objective_ratio']:.6f} | {x['N']} / 360 |")
seed_table = ['| Seed | Original ND | ND +2000 | SVM | MLOPT |', '|---|---:|---:|---:|---:|']
for seed in (1009,1013,1019):
    rows = {x['role']:x for x in audit['per_seed'] if x['seed']==seed}
    cells = [f"{rows[role]['objective_ratio']:.6f} / {rows[role]['canonical_completion_ratio']:.6f}" for role in ('neural_diving_original','neural_diving_continued','svm','mlopt')]
    seed_table.append('| '+str(seed)+' | '+' | '.join(cells)+' |')
time_table = ['| Seed | Width / parameters | Original fit (s) | Extra fit (s) | Combined (s) | Train bit error, before → after |', '|---|---:|---:|---:|---:|---:|']
for x in train:
    time_table.append(f"| {x['seed']} | {x['hidden_size']} / {x['parameter_count']:,} | {x['original_fit_wall_seconds']:.2f} | {x['additional_wall_seconds']:.2f} | {x['combined_recorded_wall_seconds']:.2f} | {100*x['initial_train_bit_error']:.3f}% → {100*x['final_train_bit_error']:.3f}% |")

text = '''The independent second stream preserves an objective-ratio advantage after the fixed additional neural-network training budget. Ours returns the original optimum on all 360 costs. Original Neural Diving has a pooled ratio of 0.960871; its fixed +2000-step control has 0.956640. All three original-ND seed means remain at least 0.0302 below Ours. These are results for this synthetic problem family, not a claim about every IP family.

There are 32 coupled bundles, 96 original binary variables, and matched K=32. Every fit uses the same 36 certified training costs for a physical seed; coordinate-fixing methods fix exactly 64 original variables. The common prefix 36 was chosen by training-rank coverage; 69 exploratory labels had previously been collected per seed. Each seed receives 120 new independent costs from the same training law. All 15 model roles and their sources were frozen before this stream was generated. The original gross-profit objective is used throughout, without subtracting the base-project returns.

'''+ '\n'.join(table)+'''

Recovery uses all 360 costs. The objective-ratio mean uses the N valid, timely returns shown above; failures are retained, and none exceeded the time budget. The pooled mean gives equal weight to each valid return.

Each cell below is **returned ratio / canonical-completion diagnostic**. Ours is 1 / 1 for all three seeds.

'''+ '\n'.join(seed_table)+'''

For a returned upgrade decision C, canonical completion adds both basic projects wherever C=0. Its gross profit is 0.5G + sum_g (xi_g − 0.5) C_g. This diagnostic preserves the returned upgrade choices and is feasible in the original problem, but can violate the method's predicted fixings. It does not replace the returned point or the primary ratio. The pooled canonical ratios are 0.962814 for original ND, 0.962460 for continued ND, 0.887723 for SVM, and 0.953409 for MLOPT. Thus, missing basic-project returns explain only part of the loss. On the 346 costs where both ND versions return valid solutions, their means are 0.961690 and 0.956717, respectively.

'''+ '\n'.join(time_table)+'''

The original ND fit retains the complete predictor, six GCN layers, 28 formulation/root-LP features, both width candidates 32/64, the 1200-step small-sample sanity gates, and the original training/selection/refit schedule. The same-label-budget adaptation uses singleton certified-optimum pools and the deterministic matched-K proposal; it does not train unused selective-mask networks. The control preserves the selected architecture and initial weights, resets Adam explicitly, and trains only the 36 labels for exactly 2000 updates (batch16,16,4; learning rate 0.001 for 1000 updates, then 0.0001). Update 2000 is the sole final checkpoint; no validation or test outcome selects it.

Extra optimization substantially lowers training error but does not improve the pooled independent objective ratio. This does not establish convergence of all possible network fits or a generalization bound. Recorded fit times exclude training-label generation and were collected with concurrent single-thread processes; they are not isolated speedup measurements. SVM model-fit times are 0.319/0.151/0.321 s; MLOPT's complete eight-grid/validation/refit times are 1.962/0.694/1.957 s. Their import/wrapper overheads are separately recorded.

Evidence: '''+link('independent check (360 references, 1800 returns, zero discrepancies)',AUDIT)+', '+link('training-control receipt',ROOT/'continuation-training-summary.json')+', '+link('architecture and complete training protocol',ROOT/'control-protocol.json')+', '+link('baseline fit-time records',EXTRAS/'learning-times.json')+'.\n'

for path,value in [(RUN/'pooled-results.json',pooled)]:
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
with (RUN/'RESULTS.md').open('x',encoding='utf-8') as f:f.write(text)
print(json.dumps({str(p.resolve()):sha(p) for p in [RUN/'pooled-results.json',RUN/'RESULTS.md']},ensure_ascii=False))
