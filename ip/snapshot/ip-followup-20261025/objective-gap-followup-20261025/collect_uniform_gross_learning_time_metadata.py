"""Collect already recorded fit intervals; do not rerun or estimate timings."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
FREEZE=ROOT/'uniform-gross-confirmation/root_all_models_freeze.json'
OUTPUT=ROOT/'uniform-gross-confirmation/recorded-learning-time-metadata.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    freeze=read(FREEZE);bindings={str(FREEZE.resolve()):sha(FREEZE)};rows=[]
    if freeze['case_count']!=12 or freeze['model_role_count']!=24:raise ValueError('Need the entire fixed cohort')
    for cid,entry in sorted(freeze['cases'].items()):
        affine_path=Path(entry['training_directory'])/'affine.json'
        affine=read(affine_path)
        if sha(affine_path)!=entry['training_files'][str(affine_path.resolve())]:
            raise ValueError('Archived affine timing is outside the all-model freeze')
        nd_path=Path(entry['model_root'])/'models'/cid/'neural_diving/complete.json'
        nd=read(nd_path)
        if sha(nd_path)!=entry['methods']['neural_diving']['files'][str(nd_path.resolve())]:
            raise ValueError('ND fit interval marker changed')
        cert_path=Path(entry['model_root'])/'training-certificates'/cid/'complete.json'
        cert=read(cert_path)
        if sha(cert_path)!=entry['methods']['ours']['files'][str(cert_path.resolve())]:
            raise ValueError('Label verification marker changed')
        for path in (affine_path,nd_path,cert_path):bindings[str(path.resolve())]=sha(path)
        rows.append(dict(case_id=cid,G=entry['G'],n=entry['n_train'],seed=entry['seed'],K=entry['K'],
            ours_archived_affine_fit_seconds=affine.get('fit_seconds'),
            neural_diving_fit_wall_seconds=nd['fit_seconds'],neural_diving_fit_cpu_seconds=nd['fit_cpu_seconds'],
            additional_label_oracle_verification_wall_seconds=cert.get('extra_oracle_verification_seconds'),
            original_training_label_generation_seconds=None,
            prior_exploratory_label_count=69 if entry['G']==32 else entry['n_train']))
    groups=[]
    for G,n in ((16,37),(20,45),(32,36),(32,69)):
        group=[row for row in rows if (row['G'],row['n'])==(G,n)]
        if len(group)!=3:raise ValueError('Never select a favorable physical seed')
        item=dict(G=G,n=n,K=G,physical_seed_count=3)
        for field in ('ours_archived_affine_fit_seconds','neural_diving_fit_wall_seconds','neural_diving_fit_cpu_seconds'):
            values=[row[field] for row in group]
            item[field]=dict(mean=statistics.mean(values),minimum=min(values),maximum=max(values)) if all(v is not None for v in values) else None
        groups.append(item)
    result=dict(status='COLLECTED_RECORDED_INTERVALS_ONLY',cells=groups,per_case=rows,source_sha256=bindings,
        collection_source_sha256=sha(__file__),
        scope='Ours: frozen producer affine-fit interval. Neural Diving: frozen training-marker fit wall/process-CPU intervals. No solve or fit was rerun.',
        excluded='Historical training-label generation was not timed. Additional label-oracle verification is separate and is not substituted for label generation.',
        concurrency='Fits ran concurrently on one-thread CPU workers. These intervals are not isolated hardware benchmarks or evidence of IP solve speedup.',
        no_invented_or_backfilled_intervals=True)
    with OUTPUT.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(OUTPUT),sha256=sha(OUTPUT),cells=groups)))


if __name__=='__main__':main()
