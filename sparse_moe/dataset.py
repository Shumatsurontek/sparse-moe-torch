import torch
from torch.utils.data import Dataset

class ToyRegressionDataset(Dataset):
    def __init__(self, num_samples=100, seq_len=16, dim=32):
        self.inputs = torch.randn(num_samples, seq_len, dim)
        self.labels = self.inputs.sum(dim=-1, keepdim=True)  # shape: (num_samples, seq_len, 1)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return {
            "inputs_embeds": self.inputs[idx],
            "labels": self.labels[idx]
        }