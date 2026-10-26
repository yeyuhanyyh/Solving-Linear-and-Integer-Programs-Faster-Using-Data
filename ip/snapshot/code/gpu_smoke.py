"""Tiny synthetic device/gradient/CPU-export check, never real case fitting."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse,json
import numpy as np
import scipy.sparse as sparse
import torch
from adapter import LinearModel
from mlopt_baseline import Config,_train_network,self_test
from neural_diving_baseline import NeuralDivingConfig,FixedBipartiteGraph,_CoverageGCN
from preference_baseline import PreferenceConfig,PreferenceNetwork,preference_loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("cuda","cpu","auto"),default="cuda")
    args=parser.parse_args()
    device=args.device
    if device=="auto":
        device="cuda" if torch.cuda.is_available() else "cpu"
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested without a usable GPU")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    decoder=self_test()
    X=np.asarray([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    weights,mean,scale=_train_network(X,np.array([0,1,0,1]),2,(8,8,8),.001,1,
        Config(device=device,batch_size=2),7)
    if not all(isinstance(w,np.ndarray) and np.isfinite(w).all() and np.isfinite(b).all() for w,b in weights):
        raise RuntimeError("MLOPT CUDA-to-CPU dense export failed")
    # The toy binary problem has optimum (1,1), and its root LP is fractional.
    model=LinearModel(sparse.csr_matrix([[1.,1.]]),np.array([1.5]),np.array([2.]),np.zeros(2),np.ones(2),np.ones(2,dtype=int))
    costs=np.array([[1.,2.],[2.,1.],[1.2,2.2],[2.1,1.1]])
    config=NeuralDivingConfig(device=device,hidden_size=32,graph_layers=6)
    graph=FixedBipartiteGraph(model,costs,normalize_rows=config.normalize_constraint_rows,
        device=device,use_root_lp_features=config.use_root_lp_features,
        root_lp_time_limit=config.root_lp_time_limit,root_lp_seed=config.seed)
    synthetic_root_lp_solves=0
    original_extract=graph.root_lp_extractor.extract
    def counted_extract(*arguments,**keywords):
        nonlocal synthetic_root_lp_solves
        synthetic_root_lp_solves+=1
        return original_extract(*arguments,**keywords)
    graph.root_lp_extractor.extract=counted_extract
    root_lp_report=graph.precompute_training_root_lp(costs)
    if root_lp_report["count"]!=4 or synthetic_root_lp_solves!=4 or not all(
            record["optimal"] and record["value_valid"] and record["dual_valid"] and record["basis_valid"]
            for record in root_lp_report["records"]):
        raise RuntimeError("Neural Diving toy root LPs lack certified primal/dual/basis features")
    net=_CoverageGCN(graph,config).to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    logits,accept=net(costs)
    labels=torch.ones((4,2),device=device)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)+accept.square().mean()
    optimizer.zero_grad();loss.backward();optimizer.step()
    if not torch.isfinite(loss) or not any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()):
        raise RuntimeError("Neural Diving sparse GPU forward/backward failed")
    graph.to("cpu");net.to("cpu")
    cpu_logits,_=net(costs)
    if cpu_logits.device.type!="cpu" or not torch.isfinite(cpu_logits).all():
        raise RuntimeError("Neural Diving CPU inference transfer failed")
    if synthetic_root_lp_solves!=4:
        raise RuntimeError("Toy cached training forwards unexpectedly solved more root LPs")
    pref=PreferenceNetwork(2,np.array([[1.,0.,1.],[0.,1.,0.]],dtype=np.float32),
                           PreferenceConfig(hidden_dim=128,attention_layers=2,device=device)).to(device)
    optimizer=torch.optim.AdamW(pref.parameters(),lr=.001)
    scores=pref(torch.tensor(X,dtype=torch.float32,device=device))
    rewards=torch.tensor([[0.,1.],[1.,0.],[0.,1.],[1.,0.]],device=device)
    loss,_,_=preference_loss(scores,rewards,.8,.2)
    optimizer.zero_grad();loss.backward();optimizer.step()
    if not torch.isfinite(loss):
        raise RuntimeError("Li GPU residual-attention gradient failed")
    pref.to("cpu").eval()
    if not torch.isfinite(pref(torch.tensor(X,dtype=torch.float32))).all():
        raise RuntimeError("Li CPU inference transfer failed")
    print(json.dumps(dict(status="synthetic_device_check_passed",device=device,
        gpu=(torch.cuda.get_device_name(0) if device=="cuda" else None),
        MLOPT_decoder_checks=decoder,synthetic_root_lp_solves=synthetic_root_lp_solves,
        root_lp_features_enabled=graph.use_root_lp_features,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        real_training_costs=0,confirmation_draws=0,real_solver_calls=0)),flush=True)


if __name__=="__main__":
    main()
