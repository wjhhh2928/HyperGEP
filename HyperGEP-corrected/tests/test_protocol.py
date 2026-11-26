import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

from hypergep.cli import ROOT, execute, main, resolved, arguments
from hypergep.data import Blocked, Transform, demo_data, file_hash, load_data, splits
from hypergep.model import (IncidenceAttention, LegacyMultiHeadFusion, TokenCrossAttentionFusion,
                            incidence_to_propagation, masked_softmax)
from hypergep.train import load_checkpoint, masked_loss, metrics, restore_base
from scripts.aggregate_results import paired_transfer, read_runs

def config(*args):
    return resolved(arguments("main").parse_args(["--dataset","demo",*args]))[0]

def fold(data):
    return next(splits(data,np.arange(len(data.ids)),"record",5,42))

def single_run(root, data, cfg, fold_data):
    number, train, test = fold_data
    execute(data,cfg,"Full_Model",42,number,train,test,root,False,data.traits)
    return next(root.rglob("run_manifest.json")).parent

class Inputs(unittest.TestCase):
    def test_align_by_identifier_and_verify_file_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/"phe.csv").write_text("sample_id,genotype_id,environment_id,Y\nb,002,10,\na,001,02,4\n")
            (directory/"geno.csv").write_text("genotype_id,m\n002,2\n001,1\n")
            (directory/"env.csv").write_text("environment_id,weather\n10,10\n02,2\n")
            paths = {"phenotypes":"phe.csv","genotypes":"geno.csv","environments":"env.csv","provenance":"p.json"}
            manifest = dict(dataset_version="fixture",origin="test",license="test",preprocessing="raw",
                            row_alignment_verified=True,label_provenance="observed",
                            environment_availability="test covariates",
                            sha256={key:file_hash(directory/value) for key,value in paths.items()
                                    if key!="provenance"})
            (directory/"p.json").write_text(json.dumps(manifest))
            cfg=dict(data_paths=paths,dataset="fixture",sample_id_column="sample_id",
                     genotype_id_column="genotype_id",environment_id_column="environment_id",
                     trait_names=["Y"])
            data = load_data(cfg,directory)
            self.assertEqual(data.ids.genotype_id.tolist(),["001","002"])
            np.testing.assert_array_equal(data.env[data.env_lookup,0],[2,10])
            np.testing.assert_array_equal(data.snp[data.snp_lookup,0],[1,2])
            self.assertTrue(np.isnan(data.y[1,0]))
            (directory/"env.csv").write_text("tampered")
            with self.assertRaises(Blocked):
                load_data(cfg,directory)

    def test_split_groups_and_coverage(self):
        data = demo_data()
        for strategy,column in (("loeo","environment_id"),("environment","environment_id"),
                                   ("genotype","genotype_id"),("site","site_id")):
            tested = []
            for _,train,test in splits(data,np.arange(len(data.ids)),strategy,3,42):
                self.assertFalse(set(train)&set(test))
                self.assertFalse(set(data.ids.iloc[train][column])&set(data.ids.iloc[test][column]))
                tested.extend(test)
            self.assertEqual(sorted(tested),list(range(len(data.ids))))

    def test_select_impute_scale_and_pca_use_training_slice(self):
        train = np.array([[0,0,np.nan],[1,0,9],[2,0,2],[3,0,5.]])
        transform = Transform.fit(train,select=2,components=1)
        state = transform.state()
        selected = {0,2}
        self.assertEqual(set(transform.columns),selected)
        self.assertAlmostEqual(transform.fill[2],16/3)
        untouched = Transform.from_state(state)
        np.testing.assert_allclose(transform.apply(train),untouched.apply(train))
        transform.apply(np.array([[50000,9999,5.]]))
        self.assertEqual(transform.state(),state)

class Models(unittest.TestCase):
    def test_legacy_and_token_attention(self):
        torch.manual_seed(15)
        env = torch.randn(3,8)
        tokens = torch.randn(3,4,8)
        old = LegacyMultiHeadFusion(8,2)
        legacy, weights = old(env,tokens,return_weights=True)
        self.assertTrue(torch.all(weights==1))
        legacy.square().sum().backward()
        self.assertEqual(old.q.weight.grad.abs().max().item(),0)
        self.assertEqual(old.k.weight.grad.abs().max().item(),0)
        new = TokenCrossAttentionFusion(8,2,.1).eval()
        mask = torch.tensor([[False,True,False,False]]*3)
        pred, w = new(env,tokens,mask,return_weights=True)
        self.assertEqual(pred.shape,(3,8))
        self.assertEqual(w.shape,(3,2,1,4))
        self.assertTrue(torch.all(w[...,1]==0))
        torch.testing.assert_close(w.sum(-1),torch.ones(3,2,1))
        for q,k in ((env+1,tokens),(env,tokens.roll(1,1))):
            _, modified = new(q,k,mask,return_weights=True)
            self.assertFalse(torch.allclose(w,modified))
        pred.square().sum().backward()
        for component in (new.query,new.key,new.value):
            self.assertGreater(component.weight.grad.norm().item(),0)

    def test_hypergraph_incidence_and_empty_neighborhood(self):
        h=np.array([[1,0,0],[1,1,0],[0,1,0],[0,0,0]])
        expected=np.array([[.5,1/(2*np.sqrt(2)),0,0],[1/(2*np.sqrt(2)),.5,1/(2*np.sqrt(2)),0],
                           [0,1/(2*np.sqrt(2)),.5,0],[0,0,0,0]])
        np.testing.assert_allclose(incidence_to_propagation(h),expected,atol=1e-10)
        mask=torch.tensor([[True,False],[False,False]])
        w=masked_softmax(torch.tensor([[4.,8.],[1.,3.]]),mask)
        torch.testing.assert_close(w,torch.tensor([[1.,0.],[0.,0.]]))
        layer=IncidenceAttention(4)
        x=torch.randn(2,4,requires_grad=True)
        y, w=layer.aggregate(x,x,torch.tensor([[1,-1],[-1,-1]]),layer.node_query,layer.edge_key)
        self.assertTrue(torch.isfinite(y).all())
        torch.testing.assert_close(w,torch.tensor([[1.,0.],[0.,0.]]))
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_masked_labels_and_undefined_correlation(self):
        prediction=torch.tensor([[1.,5.],[3.,6.]],requires_grad=True)
        target=torch.tensor([[float("nan"),2.],[3.,float("nan")]])
        loss=masked_loss(prediction,target)
        self.assertAlmostEqual(loss.item(),4.5)
        loss.backward()
        self.assertEqual(prediction.grad[0,0].item(),0)
        frame=pd.DataFrame(dict(dataset="demo",evaluation_protocol="record/inductive",experiment="main",
                                model="test",model_version="test",seed=1,fold=0,trait="Y",
                                true_value=[1.,np.nan],predicted_value=[2.,99.],label_valid=[True,False]))
        row=metrics(frame).iloc[0]
        self.assertEqual(row.valid_sample_count,1)
        self.assertEqual(row.PCC_reason,"insufficient_samples")
        self.assertTrue(np.isnan(row.PCC))
        self.assertEqual(row.MAE,1)

class FullProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_outer_labels_and_unrelated_queries_do_not_change_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=demo_data()
            spec=fold(data)
            fold_id,train,test=spec
            cfg=config("--split","environment")
            spec=next(splits(data,np.arange(len(data.ids)),"environment",5,42))
            fold_id,train,test=spec
            first=single_run(root/"a",data,cfg,spec)
            disturbed=demo_data()
            disturbed.y[test] += 99
            lookup=disturbed.env_lookup[test]
            subject=test[0]
            other_id=next(i for i in lookup if i!=disturbed.env_lookup[subject])
            disturbed.env[other_id] += 1000
            second=single_run(root/"b",disturbed,cfg,spec)
            self.assertEqual((first/"preprocessing_state.json").read_text(),
                             (second/"preprocessing_state.json").read_text())
            for path in (first/"checkpoints").glob("*.pt"):
                left=load_checkpoint(path,"cpu")["model"]
                right=load_checkpoint(second/"checkpoints"/path.name,"cpu")["model"]
                for key in left:
                    torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
            p1,p2=[pd.read_csv(path/"predictions.csv") for path in (first,second)]
            kept=p1.environment_id!=data.ids.iloc[test[lookup==other_id][0]].environment_id
            np.testing.assert_allclose(p1.loc[kept,"predicted_value"],p2.loc[kept,"predicted_value"])
            self.assertFalse(np.allclose(p1.true_value,p2.true_value,equal_nan=True))
            # Reordering and requiring multiple queries never changes a fixed query.
            base=restore_base(data,first/"checkpoints/outer_branch_0.pt",cfg,"cpu")
            np.testing.assert_allclose(base.predict(test[:1],1),base.predict(test[::-1],32)[-1:],
                                       rtol=1e-5,atol=1e-6)
            rebuilt=metrics(p1)
            saved=pd.read_csv(first/"metrics.csv")
            np.testing.assert_allclose(rebuilt.PCC,saved.PCC,equal_nan=True)

    def test_oof_manifest_integrity_reuse_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=config("--test-strategy","ensemble")
            data=demo_data()
            number,train,test=fold(data)
            run=single_run(root,data,cfg,(number,train,test))
            oof=pd.read_csv(run/"oof_predictions.csv")
            self.assertFalse(oof.duplicated(["sample_id","trait"]).any())
            self.assertEqual(len(oof),len(train)*len(data.traits))
            manifest=run/"run_manifest.json"
            previous=manifest.read_bytes()
            checkpoints=list((run/"checkpoints").glob("*.pt"))
            for inner,rows in oof.groupby("inner_fold"):
                split=pd.read_csv(run/"split_indices.tsv",sep="\t")
                refs=set(split.loc[(split.inner_fold==inner)&split.role.eq("train"),"sample_id"])
                self.assertFalse(refs&set(rows.sample_id))
                for name, group in rows.groupby("checkpoint"):
                    self.assertEqual(set(group.checkpoint_id),{file_hash(run/"checkpoints"/name)})
            from scripts.replay_checkpoint import replay
            self.assertEqual(replay(run,root,"cpu"), (len(test),len(data.traits)))
            reused=execute(data,cfg,"Full_Model",42,number,train,test,root,True,data.traits)
            self.assertEqual(reused,"COMPLETED")
            self.assertEqual(manifest.read_bytes(),previous)
            state=json.loads(previous)
            state["status"]="FAILED"
            manifest.write_text(json.dumps(state))
            self.assertEqual(execute(data,cfg,"Full_Model",42,number,train,test,root,True,data.traits),"COMPLETED")
            self.assertEqual(len(checkpoints),len(list((run/"checkpoints").glob("*.pt"))))

    def test_experiment_smokes_and_transfer_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            for exp,variant in (("main","all"),("baselines","RandomForest"),("baselines","GBLUP_Proxy"),
                                ("baselines","AdaBoost"),("baselines","MLP"),
                                ("ablation","all"),("genomic_comparators","all"),("negative_transfer","all"),("loeo","all"),
                                ("sensitivity","all"),("interpretability","all")):
                with self.subTest(experiment=exp,variant=variant):
                    status=main(exp,["--dataset","demo","--variant",variant,"--fold","0","--output-dir",tmp])
                    self.assertEqual(status,0)
            result=read_runs(Path(tmp),True)
            transfer=paired_transfer(result)
            self.assertEqual(len(transfer),3)
            np.testing.assert_allclose(transfer.Delta_PCC_MTL,transfer.MTL_PCC-transfer.STL_PCC)
            record=result[result.experiment.isin(["main","baselines"])]
            self.assertEqual(record.test_indices_hash.nunique(),1)
            self.assertTrue(list(Path(tmp).rglob("heldout_snp_gradients.npz")))
            self.assertTrue(list(Path(tmp).rglob("s6_heldout_masking.csv")))

if __name__=="__main__":
    unittest.main()
