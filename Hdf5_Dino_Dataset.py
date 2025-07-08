import h5py
import torch
from torch.utils.data import Dataset

class HDF5DinoFeatureDataset(Dataset):
    def __init__(self, h5_path, target_layer, total_layers=24):
        super().__init__()
        self.h5_path = h5_path
        self.target_layer = target_layer
        self.cond_layer = target_layer + 1 if target_layer < total_layers - 1 else None

        # 预加载样本索引
        with h5py.File(self.h5_path, "r") as f:
            self.keys = list(f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):


        key = self.keys[int(idx)]
        print(f"Loading key: {key}")

        try:
            with h5py.File(self.h5_path, "r") as f:
                x_target = torch.tensor(f[key][f"layer_{self.target_layer}"][:], dtype=torch.float32)  # [16,16,1024]
                if self.cond_layer is not None:
                    x_cond = torch.tensor(f[key][f"layer_{self.cond_layer}"][:], dtype=torch.float32)
                else:
                    x_cond = torch.zeros_like(x_target, dtype=torch.float32)

            t = torch.rand(1).squeeze(0)  # timestep
            label = torch.tensor(0)  # 占位符（无label）
        
        except Exception as e:
            print(f"[ERROR] Failed to read {key}: {e}")
            raise e
        
        t = torch.tensor(torch.rand(()).item(), dtype=torch.float32)
        label = 0

        return {
            "x_target": x_target,  # [H, W, C]
            "x_cond": x_cond,
            "t": t,
            "label": torch.tensor(label, dtype=torch.long),
            "id": key
        }
