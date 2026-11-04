"""Fixed-dimension adaptation of Xavier-style linear-SVM predictions.

OOF reliability and the original acceptance gates rank proposals. To impose
exactly rank D-K, lower-priority proposals can also be selected. This differs
explicitly from the paper's hard rejection policy. No test labels are used.
"""
from dataclasses import dataclass
from time import perf_counter
import warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from native_source import Equality,TABLE1,frequency_action,precision_threshold
from accepted_subset import EquationGraph,exact_rank,key


def classifier(seed):
    # Leave probability calibration disabled (the default). Passing the false
    # keyword itself emits a deprecation warning in recent sklearn versions.
    return make_pipeline(StandardScaler(),SVC(kernel='linear',C=1.,random_state=seed))


def metrics(labels,predicted,value):
    truth=labels==value;positive=predicted==value
    tp=int(np.sum(truth&positive))
    return dict(frequency=float(np.mean(truth)),precision=tp/max(1,int(positive.sum())),
                recall=tp/max(1,int(truth.sum())),positives=int(truth.sum()),predicted_positives=int(positive.sum()))


@dataclass
class Relation:
    i:int
    j:object
    model:object
    constant:object
    norm:float
    scores:dict
    constant_folds:int


def fit_relation(X,y,i,j,seed):
    classes=np.unique(y)
    if len(classes)==1:
        value=int(classes[0]);oof=np.full(len(y),value,dtype=np.int8)
        return Relation(i,j,None,value,0.,{v:metrics(y,oof,v) for v in (0,1)},0)
    oof=np.empty(len(y),dtype=np.int8);fallback=0
    folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='The least populated class in y has only.*',category=UserWarning)
        for train,valid in folds.split(X,y):
            present=np.unique(y[train])
            if len(present)==1:
                oof[valid]=present[0];fallback+=1
            else:
                predictor=classifier(seed);predictor.fit(X[train],y[train]);oof[valid]=predictor.predict(X[valid])
    predictor=classifier(seed);predictor.fit(X,y)
    coefficient=predictor.steps[-1][1].coef_
    norm=float(np.linalg.norm(coefficient.toarray() if hasattr(coefficient,'toarray') else coefficient))
    return Relation(i,j,predictor,None,norm,{v:metrics(y,oof,v) for v in (0,1)},fallback)


class StrictKSVM:
    def __init__(self,coordinates,temporal,D,features,report):
        self.coordinates=coordinates;self.temporal=temporal
        self.D=D;self.n_features=features;self.report=report

    def propose(self,feature,target_K):
        if isinstance(target_K,(bool,np.bool_)) or int(target_K)!=target_K:raise ValueError('Invalid K')
        target_K=int(target_K)
        if not 0<=target_K<=self.D:raise ValueError('Invalid K')
        X=np.asarray(feature,dtype=float).reshape(1,-1)
        if X.shape!=(1,self.n_features) or not np.isfinite(X).all():raise ValueError('Invalid feature')
        proposals=[]
        for relation in self.coordinates+self.temporal:
            decision=0. if relation.model is None else float(relation.model.decision_function(X)[0])
            prediction=int(relation.constant) if relation.model is None else int(relation.model.predict(X)[0])
            if relation.j is None:
                value=prediction;kind='fix1' if value else 'fix0';eq=Equality(kind,relation.i,rhs=value)
            else:
                kind='next_equal';value=1;eq=Equality(kind,relation.i,relation.j)
                frequency=relation.scores[1]['frequency']
                if prediction!=1 and frequency_action(frequency,kind)!='unconditional':continue
            score=relation.scores[value];frequency=score['frequency']
            action=frequency_action(frequency,kind)
            passed=action=='unconditional' or (action=='classify'
                and min(score['positives'],self.report['n_train']-score['positives'])>=5
                and score['precision']>=precision_threshold(frequency,kind)
                and score['recall']>=TABLE1[kind].alpha_recall)
            unanimous=frequency==1.
            margin=(decision if value else -decision)/relation.norm if relation.norm>0 else 0.
            proposals.append(dict(eq=eq,unanimous=unanimous,original_gate_passed=passed,
                precision=score['precision'],recall=score['recall'],frequency=frequency,margin=margin))
        # Keep every coordinate-fixing alternative until rank selection. A
        # temporal relation must never remove that coordinate's fixing option.
        assert sum(p['eq'].j is None for p in proposals)==self.D
        proposals.sort(key=lambda p:(0 if p['unanimous'] else 1 if p['original_gate_passed'] else 2,
            -p['precision'],-p['recall'],-p['margin'],key(p['eq'])))
        graph=EquationGraph(self.D);selected=[];ledger=[];conflicts=0;redundant=0
        for proposal in proposals:
            if graph.rank==self.D-target_K:break
            try:independent=graph.add(proposal['eq'])
            except ValueError as exc:
                if 'Inconsistent native equations' not in str(exc):raise
                conflicts+=1;continue
            if not independent:redundant+=1;continue
            selected.append(proposal['eq']);ledger.append(dict(equation=vars(proposal['eq']),
                **{k:v for k,v in proposal.items() if k!='eq'}))
        if graph.rank!=self.D-target_K:raise AssertionError('Could not reach exact target rank')
        hard=[p['eq'] for p in proposals if p['unanimous']]
        if exact_rank(hard,self.D)>self.D-target_K:raise ValueError('K contradicts the training hull')
        assert exact_rank(selected+hard,self.D)==self.D-target_K
        return selected,dict(requested_K=target_K,actual_K=self.D-graph.rank,rank=graph.rank,
            selected_count=len(selected),matched=True,forced_by_K=sum(not p['original_gate_passed'] for p in ledger),
            conflicts_skipped=conflicts,redundant_skipped=redundant,
            constant_fold_count=sum(r.constant_folds for r in self.coordinates+self.temporal),
            original_hard_gate_algorithm=False,unanimous_training_relations_preserved=True,
            selected_constraints=ledger)


def fit(features,labels,*,periods,units,seed=0):
    started=perf_counter()
    if any(isinstance(v,(bool,np.bool_)) or int(v)!=v or int(v)<1 for v in (periods,units)):
        raise ValueError('periods and units must be positive integers')
    periods,units=int(periods),int(units)
    X=np.asarray(features,dtype=float).copy();raw=np.asarray(labels)
    if not np.isin(raw,[0,1]).all():raise ValueError('Training decisions must be binary')
    y=raw.astype(np.int8,copy=True)
    if X.ndim!=2 or X.shape[1]<1 or y.shape!=(len(X),periods*units) or len(X)<5:raise ValueError('Invalid training shape')
    if not np.isfinite(X).all() or not np.isin(y,[0,1]).all():raise ValueError('Invalid training values')
    D=y.shape[1]
    coordinates=[fit_relation(X,y[:,i],i,None,seed) for i in range(D)]
    temporal=[fit_relation(X,(y[:,i]==y[:,i+units]).astype(np.int8),i,i+units,seed)
              for i in range(D-units)] if periods>1 else []
    report=dict(n_train=len(X),n_features=X.shape[1],D=D,periods=periods,units=units,seed=seed,
        constant_coordinate_count=sum(r.constant is not None for r in coordinates),
        fitted_coordinate_models=sum(r.model is not None for r in coordinates),
        fitted_temporal_models=sum(r.model is not None for r in temporal),
        constant_fold_count=sum(r.constant_folds for r in coordinates+temporal),
        original_hard_gate_algorithm=False,strict_K_adaptation=True,probability_calibration=False)
    model=StrictKSVM(coordinates,temporal,D,X.shape[1],report)
    report['fit_seconds']=perf_counter()-started
    return model,report


def predict(model,feature,*,target_K):return model.propose(feature,target_K)
