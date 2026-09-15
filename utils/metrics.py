import torch
import matplotlib.pyplot as plt

def pearson_loss(y_pred, y_true):
    vx = y_pred - torch.mean(y_pred)
    vy = y_true - torch.mean(y_true)
    return 1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8)

def visualize_trait_weights(weights, save_path="meta_attention_weights.png"):
    weights_np = weights.mean(dim=0).cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(weights_np)), weights_np)
    plt.title("Meta Attention Weights")
    plt.xlabel("Trait Index")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()