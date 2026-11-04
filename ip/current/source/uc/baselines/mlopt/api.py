"""MLOPT native strategy prediction plus a separately named matched-K wrapper."""
from __future__ import annotations
from dataclasses import asdict,replace
from itertools import product
from time import perf_counter,process_time
import pickle
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits
import mlopt_baseline as source
from scalable_strategy_bank import SparseStrategyBank
from cases import assert_same_physics
from dimension_diagnostics import assert_matched_fixings


def fit(case,validation_case,config=None):
    """External validation selects hyperparameters; final bank uses only N labels."""
    assert_same_physics(case,validation_case)
    config=source._config(config)
    started,cpu=perf_counter(),process_time()
    bank=SparseStrategyBank(case,config.active_tolerance,config.feasibility_tolerance,config.rank_tolerance)
    selected,labels,pruning=source.prune_strategies(bank,config)
    single_class=len(selected)==1
    # With one retained class every finite logit has softmax probability one;
    # cross-entropy is identically zero, including every parameter gradient.
    # A constant zero logit therefore preserves native/matched predictions and
    # avoids charging meaningless architecture/LR/epoch search to learning time.
    grid=[((),None,0)] if single_class else product(
        config.hidden_candidates,config.learning_rates,config.epoch_candidates)
    trials=[];best=None;chosen_model=None
    with threadpool_limits(limits=1):
        for hidden,lr,epochs in grid:
            begin=perf_counter()
            if single_class:
                features=np.asarray(case.train_features,dtype=float)
                mean=features.mean(axis=0);scale=features.std(axis=0)
                scale[scale<1e-12]=1.
                weights=[(np.zeros((1,features.shape[1]),dtype=np.float32),np.zeros(1,dtype=np.float32))]
                network=weights,mean,scale
            else:
                network=source._train_network(case.train_features,labels,len(selected),hidden,lr,epochs,config,config.seed)
            model=source.MLOPTModel(bank,selected,*network,config)
            correct=valid=0;gaps=[]
            for features,cost,objective in zip(validation_case.train_features,validation_case.train_costs,validation_case.train_objectives):
                pred=model.predict(features,cost)
                if pred["valid"]:
                    valid+=1;delta=abs(pred["objective"]-objective)
                    correct+=delta<=1e-6+1e-9*abs(objective)
                    gaps.append(delta/max(abs(objective),1e-12))
                else:gaps.append(1e6)
            score=(correct,valid,-float(np.mean(gaps)))
            trial=dict(hidden=list(hidden),learning_rate=lr,epochs=epochs,validation_n=validation_case.n_train,
                classifier="constant_single_class" if single_class else "trained_network",
                correct=correct,valid=valid,mean_gap_failure_penalty=float(np.mean(gaps)),seconds=perf_counter()-begin)
            trials.append(trial)
            if best is None or score>best:best,chosen_model,chosen=score,model,trial
    # Fitting is complete. Prediction uses only the model and frozen strategies;
    # do not duplicate the large shared label bank in every checkpoint.
    bank.case=SimpleNamespace(model=case.model)
    model=Baseline(chosen_model,case)
    model.manifest=dict(method="MLOPT",paper=source.PAPER,author_reference_commit=source.AUTHOR_COMMIT,
        implementation="audited-paper-port-external-validation-sparse-kkt-v2",config=asdict(config),
        training_label_count=case.n_train,validation_label_count=validation_case.n_train,
        training_sha256=case.dataset_fingerprint,validation_sha256=validation_case.dataset_fingerprint,
        selected_hyperparameters=chosen,validation_trials=trials,bank=bank.diagnostics,pruning=pruning,
        single_class_fast_path=single_class,classifier_gradient_epochs=sum(t["epochs"] for t in trials),
        single_class_equivalence=("One retained strategy has softmax probability one for every finite logit; constant zero logits preserve native ranking, decoding and matched-K fixings." if single_class else None),
        fit_seconds=perf_counter()-started,fit_cpu_seconds=process_time()-cpu,
        epochs_are_not_sample_count=True,validation_labels_used_in_final_bank=False,test_information_used=False,
        native_dimension_K=None,matched_dimension_scope="commitment coordinates only",
        native_algorithm="complete integer strategy plus active constraints; top-k equality KKT decoding",
        adaptations=["bounded shared label budget instead of Good-Turing exploration", "external common validation",
            "existing audited pruning and network grid", "sparse unregularized normal equations; failed decoders are retained as failures"],
        matched_adaptation="native decoded winner supplies anchor; predicted top-k probability agreement ranks u fixings; reoptimize original MILP",
        training_label_collection_included=False)
    return model


class Baseline:
    def __init__(self,model,case):self.model=model;self.case=case;self.manifest={}

    def __getstate__(self):
        state=dict(self.__dict__)
        state["case"]=SimpleNamespace(model=self.case.model,
            commitment_indices=self.case.commitment_indices,affine=self.case.affine,family=self.case.family)
        return state

    def predict_native(self,cost,features):return self.model.predict(features,cost)

    def predict_restriction(self,cost,features=None,target_K=None):
        if features is None:raise ValueError("Explicit objective features are required")
        started=perf_counter();m=self.model;common=self.case.commitment_indices
        K=int(self.case.affine["rank"] if target_K is None else target_K)
        pred=m.predict(features,cost)
        if not pred["valid"]:
            return dict(fixed=None,extra_equalities=None,prediction_seconds=perf_counter()-started,
                        diagnostics=dict(status="native_no_feasible_anchor",native=pred["diagnostics"]))
        values=((np.asarray(features)-m.mean)/m.scale).astype(np.float32)
        for i,(weight,bias) in enumerate(m.weights):
            values=weight@values+bias
            if i+1<len(m.weights):values=np.maximum(values,0)
        ranked=np.argsort(-values,kind="stable")[:min(m.config.n_best,len(m.selected))]
        logits=values[ranked].astype(float);prob=np.exp(logits-logits.max());prob/=prob.sum()
        loc={int(j):i for i,j in enumerate(m.bank.integer_indices)}
        ids=[loc[int(j)] for j in common]
        candidates=np.asarray([m.bank.strategies[int(m.selected[j])].integer_values[ids] for j in ranked])
        anchor=np.rint(pred["x"][common]).astype(int)
        agreement=np.sum(prob[:,None]*(candidates==anchor),axis=0)
        order=np.lexsort((common,-agreement));selected=order[:len(common)-K]
        fixed={int(common[j]):int(anchor[j]) for j in selected}
        dims=assert_matched_fixings(self.case,fixed,K)
        return dict(fixed=fixed,extra_equalities=None,prediction_seconds=perf_counter()-started,
            diagnostics=dict(method="MLOPT-anchor-matchedK",native_strategy_id=pred["strategy_id"],
                confidence="top-k softmax probability weighted agreement with native decoded winner",
                native_candidates=pred["diagnostics"],dimensions=dims))

    def save(self,path):
        with open(path,"wb") as f:pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path,"rb") as f:return pickle.load(f)
