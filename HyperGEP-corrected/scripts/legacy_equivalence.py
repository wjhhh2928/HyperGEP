"""Numerically compare local monolithic and modular legacy components."""
import argparse
import ast
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

def extract(file,names,namespace=None):
    tree=ast.parse(file.read_text(encoding="utf-8-sig"),filename=file.name)
    selected=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    context=dict(torch=torch,nn=nn,F=F,np=np,Dataset=Dataset)
    context.update(namespace or {})
    exec(compile(ast.Module(body=selected,type_ignores=[]),file.name,"exec"),context)
    return context

def compare(root):
    original=root/"train_3hgat_cnn_5zhe_xin.py"
    components={
        "dataset/dataset.py":["FusionDataset"],"models/env_encode.py":["HGATLayer","EnvHGAT"],
        "models/snp_encode.py":["SNPEncoderCNN"],"models/fusion.py":["MultiHeadFusion","MetaTraitFusion"],
        "utils/graph.py":["construct_H_with_KNN_from_distance","generate_G_from_H"],
        "utils/metrics.py":["pearson_loss"],
    }
    monolithic=extract(original,{name for group in components.values() for name in group})
    modular={}
    for path,names in components.items():
        definitions=extract(root/path,names)
        modular.update({name:definitions[name] for name in names})
    torch.manual_seed(42)
    torch.set_num_threads(1)
    env,snp,traits=torch.randn(8,6),torch.randn(8,96),torch.randn(8,3)
    h=torch.eye(8)+torch.eye(8).roll(1,1)
    inputs={"HGATLayer":((6,8),(env,h)),"EnvHGAT":((6,16,8,4),(env,h)),
            "SNPEncoderCNN":((96,8),(snp,)),"MultiHeadFusion":((6,8,16,4),(env,torch.randn(8,8))),
            "MetaTraitFusion":((3,),(traits[:,:,None],))}
    results={}
    for name,(args,forward) in inputs.items():
        a,b=monolithic[name](*args).eval(),modular[name](*args).eval()
        try:
            b.load_state_dict(a.state_dict())
            with torch.no_grad():
                lhs,rhs=a(*forward),b(*forward)
            torch.testing.assert_close(lhs,rhs,rtol=0,atol=0)
            results[name]="VERIFIED"
        except (AssertionError,RuntimeError) as exc:
            results[name]="DIFFERENT "+str(exc).splitlines()[0]
    args=(env.numpy(),snp.numpy(),traits.numpy(),1)
    a,b=monolithic["FusionDataset"](*args),modular["FusionDataset"](*args)
    for idx in range(len(a)):
        torch.testing.assert_close(a[idx],b[idx],rtol=0,atol=0)
    results["FusionDataset"]="VERIFIED"
    distance=np.abs(np.arange(8)[:,None]-np.arange(8)[None,:]).astype(float)
    for name,args in (("pearson_loss",(traits[:,0],traits[:,1])),
                      ("construct_H_with_KNN_from_distance",(distance,3)),("generate_G_from_H",(h.numpy(),))):
        np.testing.assert_allclose(np.asarray(monolithic[name](*args)),np.asarray(modular[name](*args)),atol=0,rtol=0)
        results[name]="VERIFIED"
    # Keep the exact legacy forward and check whether its Q/K actually learn.
    fusion=monolithic["MultiHeadFusion"](6,8,16,4)
    saved=[]
    hook=fusion.attn.register_forward_hook(lambda model,inputs,out:saved.append(out[1].detach()))
    fusion(env,torch.randn(8,8)).square().sum().backward()
    hook.remove()
    assert torch.all(saved[0]==1)
    for name in ("q","k"):
        assert getattr(fusion,name).weight.grad.abs().max().item()==0
    results["legacy_attention"]="VERIFIED: weights all one and Q/K gradient exactly zero"
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument("local_sources",type=Path)
    args=p.parse_args()
    for name,status in compare(args.local_sources).items():
        print(name,status,sep="\t")

if __name__=="__main__":
    main()
