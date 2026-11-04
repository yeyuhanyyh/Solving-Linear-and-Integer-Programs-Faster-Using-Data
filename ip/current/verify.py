"""Recompute Tables 3--5 from saved observations and fitting records."""
from collections import defaultdict
from pathlib import Path
import argparse, csv, gzip, hashlib, json, math, statistics

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def check_manifest():
    manifest = read(ROOT/'manifest.json')
    for name, expected in manifest['files'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Path outside release: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed release file: '+name)
    return len(manifest['files'])

def check_fit_evidence(fits):
    def compressed(name):
        with gzip.open(ROOT/'timing'/name,'rt',encoding='utf-8-sig') as stream:
            return json.load(stream)
    cpu=compressed('correlated_cpu.json.gz')['fits']
    uc=compressed('uc_derivation.json.gz')
    for row in fits:
        p,n,m,s=row['problem'],row['n'],row['method'],row['seed']
        if p=='grouped':
            key={'Ours':'ours','SVM':'svm','MLOPT':'mlopt','Neural Diving':'neural_diving','Preference Learning':'preference'}[m]
            raw=compressed(f'grouped/{s}-{key}.json.gz')
            values=[q['headline_fit_seconds'] for q in raw['repetitions']]
        elif m=='MLOPT':
            key='case14-n512' if p=='uc' else f'knapsack-{s}-n{n}'
            raw=compressed(f'mlopt/{key}.json.gz')
            values=[q['learning_fit_seconds'] for q in raw['repetitions']]
        elif p=='uc':
            if m=='SVM':values=[q['learning_fit_seconds'] for q in compressed('uc_svm.json.gz')['fit_records']]
            elif m=='Neural Diving':
                q=uc['timing_derivation']['neural_diving']
                values=[q['epoch_selection_seconds']+q['final_refit_seconds']-q['validation_subIP_wall_seconds']]
            elif m=='Preference Learning':
                q=uc['timing_derivation']['preference_learning']
                values=[q['original_internal_fit_seconds']-q['development_all_decoder_wall_seconds']-q['final_all_decoder_wall_seconds']]
            else:values=[q['wall_seconds'] for q in uc['ours_fit_repetitions'] if q['phase']=='warm']
        elif m in ('Ours','SVM'):
            q=next(q for q in cpu if q['n_train']==n and q['seed']==s)
            values=q[m+'_repetitions']
        else:
            key='neural_diving_repaired' if m=='Neural Diving' else 'preference_learning'
            q=compressed(f'correlated/{s}-n{n}-{key}.json.gz')
            values=[q['epoch_selection_seconds']+q['final_refit_seconds']] if m=='Neural Diving' else [q['fit_seconds']]
        if not math.isclose(statistics.mean(values),statistics.mean(row['fit_seconds']),rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Fitting summary differs from its original measurements: '+str((p,n,m,s)))

def summarize(observations, fits):
    groups = defaultdict(list)
    times = defaultdict(list)
    for q in observations:
        groups[q['problem'], q['n'], q['method']].append(q)
    for q in fits:
        times[q['problem'], q['n'], q['method']].append(statistics.mean(q['fit_seconds']))
    result = {}
    for key, records in groups.items():
        problem, n, method = key
        expected = 256 if problem == 'uc' else 360 if problem == 'grouped' else 300
        if len(records) != expected or len({(r['seed'], r['index']) for r in records}) != expected:
            raise ValueError('Missing or duplicate tests: '+str(key))
        by_seed = defaultdict(list)
        for q in records:
            by_seed[q['seed']].append(q)
            recovered = q['valid'] and q['value'] is not None and abs(q['value']-q['reference']) <= 1e-6+1e-9*abs(q['reference'])
            if recovered != q['recovery']:
                raise ValueError('Incorrect recovery flag: '+str(key))
        kvals = []
        for seed, entries in sorted(by_seed.items()):
            if {r['index'] for r in entries} != set(range(len(entries))):
                raise ValueError('Nonconsecutive test indices')
            ranks = {r['K'] for r in entries}
            if len(ranks) != 1:
                raise ValueError('Dimension changes across tests')
            kvals.append(next(iter(ranks)))
        valid = [q for q in records if q['valid']]
        if problem == 'grouped':
            ratio = statistics.mean(q['value']/q['reference'] for q in valid)
        else:
            if len(valid) != expected:
                raise ValueError('Relative-ratio cohort must contain every planned return')
            denominator = math.fsum(q['reference']-q['anchor'] for q in valid)
            if denominator == 0:
                raise ValueError('Undefined relative ratio: zero anchor improvement')
            ratio = math.fsum(q['value']-q['anchor'] for q in valid)/denominator
        result[key] = dict(problem=problem,n=n,method=method,
            K=str(kvals[0]) if len(set(kvals))==1 else ','.join(map(str,kvals)),
            recovery_percent=100*sum(q['recovery'] for q in records)/expected,
            objective_ratio=ratio,learning_fit_seconds=statistics.mean(times[key]),
            planned_tests=expected,valid_ratio_tests=len(valid))
    return result

def verify(output=None):
    files = check_manifest()
    observations = [json.loads(s) for s in (ROOT/'observations.jsonl').read_text().splitlines()]
    fitting = read(ROOT/'learning_records.json')
    check_fit_evidence(fitting)
    result = summarize(observations, fitting)
    expected = list(csv.DictReader((ROOT/'results.csv').open(encoding='utf-8')))
    if len(expected) != 30 or len(result) != 30:
        raise ValueError('Expected all 30 learned-method rows')
    for r in expected:
        actual = result[r['problem'], int(r['n']), r['method']]
        for name in ['recovery_percent','objective_ratio','learning_fit_seconds']:
            if not math.isclose(actual[name],float(r[name]),rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError(f'{r["problem"]}/{r["n"]}/{r["method"]}: {name} differs: {actual[name]} vs {r[name]}')
        for name in ['K','planned_tests','valid_ratio_tests']:
            if str(actual[name]) != str(r[name]):
                raise ValueError('Count/dimension mismatch: '+name)
    # All methods see the same references/anchor and match the same K.
    shared = defaultdict(list)
    for q in observations:
        shared[q['problem'],q['n'],q['seed'],q['index']].append(q)
    for key, entries in shared.items():
        if len(entries) != 5 or len({q['K'] for q in entries}) != 1:
            raise ValueError('Method coverage or matched K differs: '+str(key))
        for field in ['reference','anchor']:
            if any(not math.isclose(q[field],entries[0][field],rel_tol=1e-10,abs_tol=1e-7) for q in entries):
                raise ValueError('Different test cohort/anchor: '+str(key))
    if output:
        output = Path(output).resolve()
        output.mkdir(parents=True,exist_ok=False)
        for table,problem in [('3','uc'),('4','grouped'),('5','correlated')]:
            rows=[result[r['problem'],int(r['n']),r['method']] for r in expected if r['problem']==problem]
            with (output/f'table{table}.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    return dict(status='PASS',files=files,rows=len(result),test_records=len(observations),
        same_K_and_test_costs=True,all_learning_times_recomputed=True,
        note='Arithmetic reproduction of saved results. This command does not retrain or solve IPs.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    print(json.dumps(verify(p.parse_args().output),indent=2))
