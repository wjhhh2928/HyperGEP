import torch.nn as nn

class SNPEncoderCNN(nn.Module):
    def __init__(self, input_len=1024, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.encoder(x.unsqueeze(1))