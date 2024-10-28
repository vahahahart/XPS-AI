from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class XPSDataset(Dataset):
    def __init__(self, path, device='cpu'):
        super().__init__()

        self.device = torch.device(device)
        self.data = []

        for f in Path(path).iterdir():
            array = np.loadtxt(f, delimiter=',')
            self.data.append(array)
        
    def __getitem__(self, index):
        array = self.data[index]

        x = array[:, 0]
        x = torch.tensor(x, dtype=torch.float32, device=self.device)

        x_log = array[:, 1]
        x_log = torch.tensor(x_log, dtype=torch.float32, device=self.device)

        peak_mask = array[:, 2]
        peak_mask = torch.tensor(peak_mask, dtype=torch.float32, device=self.device)

        max_mask = array[:, 3]
        max_mask = torch.tensor(max_mask, dtype=torch.float32, device=self.device)
        
        return torch.stack((x, x_log), dim=0), peak_mask, max_mask
    
    def __len__(self):
        return len(self.data)        
