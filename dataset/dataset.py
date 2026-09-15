import torch
from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self, env_feat, snp_feat, trait, target_index):
        self.env_feat = torch.tensor(env_feat, dtype=torch.float32)
        self.snp_feat = torch.tensor(snp_feat, dtype=torch.float32)
        self.trait = torch.tensor(trait[:, target_index], dtype=torch.float32)

    def __len__(self):
        return len(self.env_feat)

    def __getitem__(self, idx):
        return idx, self.env_feat[idx], self.snp_feat[idx], self.trait[idx]