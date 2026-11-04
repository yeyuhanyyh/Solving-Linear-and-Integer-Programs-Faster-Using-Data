"""Secondary K control: delete native accepted SVM equations, never add any.

This module neither fits models nor solves problems. The caller must retain the
entire original model and domains. K refers only to the imposed commitment
affine space, before the original physical constraints are intersected with it.
"""
from time import perf_counter
import numpy as np

POLICY='unanimous-first,precision,normalized-margin,frequency,canonical-id-v1'

def key(eq):return (str(eq.kind),int(eq.i),-1 if eq.j is None else int(eq.j),int(eq.rhs))

class EquationGraph:
    """Exact affine rank/consistency for unary fixes and equal-coordinate rows."""
    def __init__(self,D):
        self.D=D;self.parent=list(range(D));self.fixed={};self.rank=0
    def find(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i
    def add(self,eq):
        if isinstance(eq.i,bool) or int(eq.i)!=eq.i or not 0<=int(eq.i)<self.D:raise ValueError('Invalid coordinate')
        i=int(eq.i);a=self.find(i)
        if eq.j is None:
            if eq.rhs not in (0,1):raise ValueError('Fix must be binary')
            if a in self.fixed:
                if self.fixed[a]!=eq.rhs:raise ValueError('Inconsistent native equations')
                return False
            self.fixed[a]=int(eq.rhs);self.rank+=1;return True
        if isinstance(eq.j,bool) or int(eq.j)!=eq.j or not 0<=int(eq.j)<self.D or eq.rhs!=0:raise ValueError('Invalid equality')
        b=self.find(int(eq.j))
        if a==b:return False
        fa,fb=self.fixed.get(a),self.fixed.get(b)
        if fa is not None and fb is not None and fa!=fb:raise ValueError('Inconsistent native equations')
        independent=fa is None or fb is None
        self.parent[b]=a;self.fixed.pop(b,None)
        if fa is not None or fb is not None:self.fixed[a]=fa if fa is not None else fb
        self.rank+=int(independent)
        return independent

def exact_rank(equalities,D):
    graph=EquationGraph(D)
    for eq in equalities:graph.add(eq)
    return graph.rank

def normalized_margin(record,feature):
    """Distance in the classifier's existing standardized feature coordinates."""
    classifier=record.classifier
    if classifier is None:return None
    if not hasattr(classifier,'steps'):return None
    last=classifier.steps[-1][1]
    if getattr(last,'kernel',None)!='linear' or not hasattr(last,'coef_'):return None
    coef=last.coef_.toarray() if hasattr(last.coef_,'toarray') else np.asarray(last.coef_,dtype=float)
    norm=float(np.linalg.norm(coef))
    if not np.isfinite(norm) or norm<=0:return None
    decision=np.asarray(classifier.decision_function(np.asarray(feature,dtype=float).reshape(1,-1)),dtype=float).ravel()
    if decision.size!=1 or not np.isfinite(decision[0]):raise ValueError('Invalid native margin')
    return float(decision[0]/norm)

def accepted_scores(equalities,records,feature):
    by_key={key(record.equality):record for record in records}
    if len(by_key)!=len(records):raise ValueError('Duplicate native candidate IDs')
    scores={}
    for eq in equalities:
        record=by_key.get(key(eq))
        if record is None or record.status not in ('unconditional','accepted_classifier'):
            raise ValueError('Selected equation was not accepted by native gates')
        frequency=float(record.frequency)
        if not np.isfinite(frequency) or not 0<=frequency<=1:raise ValueError('Invalid native frequency')
        precision=frequency if record.status=='unconditional' else float(record.oof_precision)
        if not np.isfinite(precision) or not 0<=precision<=1:raise ValueError('Invalid native precision')
        margin=normalized_margin(record,feature) if record.status=='accepted_classifier' else None
        scores[key(eq)]=dict(unanimous=record.status=='unconditional' and frequency==1.,
            precision=precision,normalized_margin=margin,frequency=frequency,status=record.status)
    return scores

def relax_equalities(equalities,D,target_K,scores):
    """Return original equations unchanged, or an independent accepted subset."""
    if isinstance(target_K,(bool,np.bool_)) or int(target_K)!=target_K or not 0<=int(target_K)<=D:raise ValueError('Invalid target K')
    target_K=int(target_K);native=list(equalities);ids=[key(e) for e in native]
    if len(set(ids))!=len(ids) or set(ids)!=set(scores):raise ValueError('Native equation/score identity mismatch')
    native_rank=exact_rank(native,D);native_K=D-native_rank;target_rank=D-target_K
    info=dict(policy=POLICY,common_D=D,requested_K=target_K,K_native=native_K,
        native_rank=native_rank,native_count=len(native),native_equation_ids=ids,
        accepted_subset_only=True,rejected_equations_added=0,original_physics_and_domains_retained_by_caller=True,
        dimension_scope='Imposed commitment affine dimension before original physics')
    if native_K>=target_K:
        info.update(actual_K=native_K,imposed_rank=native_rank,matched=native_K==target_K,
            action='unchanged_equal' if native_K==target_K else 'unchanged_native_K_larger',
            selected_equation_ids=ids,removed_equation_ids=[],hard_unanimous_preserved=True)
        return native,info
    hard=[e for e in native if scores[key(e)]['unanimous']]
    hard_rank=exact_rank(hard,D)
    if hard_rank>target_rank:raise ValueError('Target K conflicts with unanimous relations from the same training affine hull')
    def priority(eq):
        s=scores[key(eq)];margin=s['normalized_margin']
        return (-int(s['unanimous']),-s['precision'],-(0. if margin is None else margin),-s['frequency'],key(eq))
    ordered=sorted(native,key=priority);graph=EquationGraph(D);selected=[];ledger=[]
    for eq in ordered:
        independent=False
        if graph.rank<target_rank:independent=graph.add(eq)
        keep=independent
        if keep:selected.append(eq)
        ledger.append(dict(equation_id=key(eq),score=scores[key(eq)],selected=keep))
    if graph.rank!=target_rank:raise AssertionError('Could not reach target rank from native accepted equations')
    selected_ids={key(e) for e in selected}
    # Every unanimous relation must still follow from the selected rows.
    rank_with_hard=exact_rank(selected+hard,D)
    if rank_with_hard!=target_rank:raise AssertionError('A unanimous training relation was relaxed')
    info.update(actual_K=target_K,imposed_rank=target_rank,matched=True,action='relaxed_native_subset',
        unanimous_rank=hard_rank,hard_unanimous_preserved=True,
        selected_equation_ids=[key(e) for e in selected],removed_equation_ids=[i for i in ids if i not in selected_ids],
        priority_ledger=ledger)
    return selected,info

def predict_restriction(svm,cost,*,features=None,target_K):
    """Use the fitted native SVMBaseline without modifying it or its metadata."""
    started=perf_counter()
    if svm.config.mode!='native' or svm.native_model is None:raise ValueError('Require fitted native SVMBaseline')
    value=np.asarray(cost,dtype=float)
    if value.shape!=(svm.n_variables,) or not np.isfinite(value).all():raise ValueError('Invalid objective')
    if features is None:
        if not callable(svm._features):raise ValueError('Explicit frozen features required')
        feature=np.asarray(svm._features(value[None,:]),dtype=float)
    else:
        feature=np.asarray(features,dtype=float)
        if feature.ndim==1:feature=feature[None,:]
    if feature.shape!=(1,svm.n_features) or not np.isfinite(feature).all():raise ValueError('Invalid feature map')
    native,native_info=svm.native_model.predict_equalities(feature[0])
    scores=accepted_scores(native,svm.native_model.records_,feature[0])
    common=np.asarray(svm.common_indices,dtype=int);D=len(common)
    if len(set(common.tolist()))!=D or np.any(common<0) or np.any(common>=svm.n_variables):raise ValueError('Invalid common coordinate map')
    chosen,info=relax_equalities(native,D,target_K,scores)
    fixed={};rows=[]
    for eq in chosen:
        if eq.j is None:fixed[int(common[eq.i])]=int(eq.rhs)
        else:
            row=np.zeros(svm.n_variables);row[common[eq.i]]=1.;row[common[eq.j]]-=1.;rows.append(row)
    extra=None if not rows else (np.asarray(rows),np.zeros(len(rows)))
    elapsed=perf_counter()-started
    info.update(method='SVM native accepted-equation relaxation for K control',is_native_algorithm=False,
        native_primary_unchanged=True,native_diagnostics=native_info,original_Xavier_acceptance_gates_used=True,
        prediction_seconds=elapsed,solver_called=False,test_optimum_used=False,
        fitting_performed=False,full_fallback_used=False)
    return dict(fixed=fixed,extra_equalities=extra,
        additional_rows=None if extra is None else (extra[0],extra[1],extra[1].copy()),
        diagnostic=info,prediction_seconds=elapsed)
