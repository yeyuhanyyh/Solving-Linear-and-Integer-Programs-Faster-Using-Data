from pathlib import Path
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone

research=Path('D:/C盘迁移归档/桌面大文件夹/research')
repo=research/'overleaf-iclr-lp-revision-20260920'
audit=research/'lp-iclr-revision-audit-20260920/ip-complete-better-settings-20261025'
def git(*args):
    return subprocess.check_output(['git',*args],cwd=repo).decode().strip()
assert git('rev-parse','HEAD')==git('rev-parse','origin/master')
files=[]
def visit(path):
    if path in files:
        return
    files.append(path)
    text=path.read_text(encoding='utf-8-sig')
    for name in re.findall(r'\\(?:input|include)\{([^}]+)\}',text):
        child=path.parent/name
        if not child.suffix:
            child=child.with_suffix('.tex')
        assert child.exists(),str(child)
        visit(child)
visit(repo/'neurips26_camera_ready.tex')
contents={str(path.relative_to(repo)):path.read_text(encoding='utf-8-sig') for path in files}
stale_tokens=['tab:ip-block-cost-full','tab:ip-block-cost-low','Across all six tested settings','all six settings','uniform-cost UC','half demand','3,424','62,760']
findings=[]
for name,text in contents.items():
    for lineno,line in enumerate(text.splitlines(),1):
        for token in stale_tokens:
            if token in line:
                findings.append({'file':name,'line':lineno,'token':token})
        if re.search(r'\bTables?[\s~]+(?:5|6)\b',line):
            findings.append({'file':name,'line':lineno,'token':'literal Table5/6'})
assert not findings,findings
table_records=[]
for name,text in contents.items():
    for table in re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',text,re.S):
        table_records.append({'file':name,'labels':re.findall(r'\\label\{([^}]+)\}',table)})
aux=(audit/'paper/neurips26_camera_ready.aux').read_text(encoding='utf-8-sig')
for label,number in [('tab:ip-matched-uc','3'),('tab:ip-matched-knapsack','4')]:
    assert f'\\newlabel{{{label}}}{{{{{number}}}' in aux,label
result={
    'status':'PASS','checked_utc':datetime.now(timezone.utc).isoformat(),
    'commit':git('rev-parse','HEAD'),'active_source_files':list(contents),
    'source_sha256':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    'stale_patterns_checked':stale_tokens+['literal Table5/6'],
    'stale_findings':findings,'active_tables':table_records,
    'verified_IP_numbering':{'tab:ip-matched-uc':3,'tab:ip-matched-knapsack':4},
    'source_changes_required':False,
    'archived_old_results_untouched':True
}
(audit/'current-table-reference-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'],'active_source_files':list(contents),'stale_findings':findings,'verified_IP_numbering':result['verified_IP_numbering']},ensure_ascii=True,indent=2))
