import torch
from torch.utils.data import Dataset
import numpy as np

class CryptoDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]
