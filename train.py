import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr

from dataset.dataset import FusionDataset
from models.env_encode import EnvHGAT
from models.snp_encode import SNPEncoderCNN
from models.fusion import MultiHeadFusion, MetaTraitFusion
from utils.graph import construct_H_with_KNN_from_distance, generate_G_from_H
from utils.metrics import pearson_loss, visualize_trait_weights

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default="/mnt/mydisk/zyq-mnt/G-E-Phes/env-geno/train_data/new_data/env_new2_3000.npy")
    parser.add_argument("--snp_path", type=str, default="/mnt/mydisk/zyq-mnt/G-E-Phes/env-geno/train_data/new_data/gene_new2_3000.npy")
    parser.add_argument("--phe_path", type=str, default="/mnt/mydisk/zyq-mnt/G-E-Phes/env-geno/train_data/new_data/phe_new2_3000_star.csv")
    #default -1，Represents automatically running all traits
    parser.add_argument("--target_idx", type=int, default=-1, help="Target trait index. Set to -1 to run all traits sequentially.")
    parser.add_argument("--k_neig", type=int, default=9)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--meta_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()

def main():
    args = parse_args()
    
    
    print(">>> Loading Data...")
    env = np.load(args.env_path)
    snp = np.load(args.snp_path)
    phe_df = pd.read_csv(args.phe_path)
    phe = phe_df.select_dtypes(include=[np.number]).values
    num_traits = phe.shape[1]

    env = StandardScaler().fit_transform(env)
    snp = StandardScaler().fit_transform(snp)
    snp_len = snp.shape[1]

    
    print(">>> Constructing Hypergraph...")
    H = construct_H_with_KNN_from_distance(euclidean_distances(env), args.k_neig)
    G_full = generate_G_from_H(H)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    print("\n>>> Phase 1: Pretraining Encoders (Feature Extraction) <<<")
    preds_all = []
    for i in range(num_traits):
        print(f"\n--- Pretraining Trait {i} ---")
        train_idx, val_idx = train_test_split(np.arange(len(env)), test_size=0.2, random_state=42)
        train_loader = DataLoader(FusionDataset(env[train_idx], snp[train_idx], phe[train_idx], i), batch_size=32, shuffle=True)
        
        env_model = EnvHGAT(env.shape[1]).to(device)
        snp_model = SNPEncoderCNN(snp_len).to(device)
        fusion = MultiHeadFusion(64, 128).to(device)
        opt = torch.optim.Adam(list(env_model.parameters()) + list(snp_model.parameters()) + list(fusion.parameters()), lr=args.lr)

        for epoch in range(args.pretrain_epochs):
            env_model.train(); snp_model.train(); fusion.train()
            epoch_loss = 0
            for idx, env_x, snp_x, y in train_loader:
                G = G_full[train_idx[idx.numpy()]][:, train_idx[idx.numpy()]].to(device)
                env_x, snp_x, y = env_x.to(device), snp_x.to(device), y.to(device)
                pred = fusion(env_model(env_x, G), snp_model(snp_x))
                loss = nn.MSELoss()(pred, y) + 0.2 * pearson_loss(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()
                
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Trait {i} | Epoch {epoch+1:03d}/{args.pretrain_epochs} | Loss: {epoch_loss / len(train_loader):.4f}")

        
        fusion.eval(); env_model.eval(); snp_model.eval()
        with torch.no_grad():
            all_preds = []
            for start in range(0, len(env), 32):
                idx = np.arange(start, min(start+32, len(env)))
                G = G_full[idx][:, idx].to(device)
                env_x = torch.tensor(env[idx], dtype=torch.float32).to(device)
                snp_x = torch.tensor(snp[idx], dtype=torch.float32).to(device)
                pred = fusion(env_model(env_x, G), snp_model(snp_x))
                all_preds.append(pred.cpu())
            preds_all.append(torch.cat(all_preds).unsqueeze(1))

    fused = torch.cat(preds_all, dim=1).unsqueeze(-1).to(device)

    
    target_indices = range(num_traits) if args.target_idx == -1 else [args.target_idx]

   
    print("\n>>> Phase 2: Meta-Fusion 5-Fold Cross Validation <<<")
    
    for t_idx in target_indices:
        print(f"\n{'='*45}")
        print(f"🚀 Evaluating Target Trait [{t_idx}]")
        print(f"{'='*45}")
        
        y_true = torch.tensor(phe[:, t_idx], dtype=torch.float32).to(device)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        metrics_per_fold = []
        best_weights_global = None  
        highest_pcc = -1
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(fused.shape[0]))):
            fold_id = fold + 1
            meta = MetaTraitFusion(num_traits).to(device)
            opt = torch.optim.Adam(meta.parameters(), lr=args.lr)

            fused_train, y_train = fused[train_idx], y_true[train_idx]
            fused_test, y_test = fused[test_idx], y_true[test_idx]

            best_pcc = -1
            best_r2 = -1e9
            best_mae = -1
            best_rmse = -1
            best_weights_fold = None
            
            for epoch in range(args.meta_epochs):
                meta.train()
                pred_tr, _ = meta(fused_train)
                loss = nn.MSELoss()(pred_tr, y_train)
                opt.zero_grad(); loss.backward(); opt.step()

                meta.eval()
                with torch.no_grad():
                    pred_te, final_weights = meta(fused_test)
                    y_np = y_test.cpu().numpy()
                    y_pred_np = pred_te.cpu().numpy()
                    
                    pear = pearsonr(y_np, y_pred_np)[0]
                    r2 = r2_score(y_np, y_pred_np)
                    
                    if pear > best_pcc:
                        best_pcc = pear
                        best_r2 = r2
                        best_mae = mean_absolute_error(y_np, y_pred_np)
                        best_rmse = np.sqrt(mean_squared_error(y_np, y_pred_np))
                        best_weights_fold = final_weights

            metrics_per_fold.append({
                "fold": fold_id, "R2": best_r2, "Pearson": best_pcc, 
                "MAE": best_mae, "RMSE": best_rmse
            })
            print(f"  [Fold {fold_id}] PCC: {best_pcc:.4f} | R2: {best_r2:.4f} | MAE: {best_mae:.4f} | RMSE: {best_rmse:.4f}")
            
            
            if best_pcc > highest_pcc:
                highest_pcc = best_pcc
                best_weights_global = best_weights_fold

        
        cols = ["R2", "Pearson", "MAE", "RMSE"]
        M = np.array([[m[c] for c in cols] for m in metrics_per_fold], dtype=float)
        mean, std = M.mean(axis=0), M.std(axis=0, ddof=1)
        
        df_metrics = pd.DataFrame(metrics_per_fold)
        df_metrics.loc['mean'] = ["mean"] + list(mean)
        df_metrics.loc['std']  = ["std"] + list(std)
        
        
        csv_name = f"meta_fusion_5fold_metrics_trait_{t_idx}.csv"
        png_name = f"meta_attention_weights_trait_{t_idx}.png"
        
        df_metrics.to_csv(csv_name, index=False, sep=',')
        if best_weights_global is not None:
            visualize_trait_weights(best_weights_global, png_name)
        
        print(f" Trait {t_idx} Complete. Saved to '{csv_name}' and '{png_name}'.")

    print("\n All requested traits have been processed successfully!")

if __name__ == "__main__":
    main()
