from Hdf5_Dino_Dataset import HDF5DinoFeatureDataset
from torch.utils.data import DataLoader

dataset = HDF5DinoFeatureDataset("features/mini_subset_for_validation.h5", target_layer=23)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

for i, batch in enumerate(loader):
    print(f"[Batch {i}] x_target: {batch['x_target'].shape}, x_cond: {batch['x_cond'].shape}, t: {batch['t']}")
    if i == 3:
        break
