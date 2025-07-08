import os
import h5py
import torch
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image


#离线提取DINO特征图，并且HDF格式压缩存储
#目前丢掉了cls token，后续估计得加回来

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = ImageFolder("/home/user/user/data/mini_subset", transform=transform)
os.makedirs("features", exist_ok=True)

hdf5_path = "features/mini_subset_for_validation.h5"
with h5py.File(hdf5_path, "w") as h5f:
    for i, (img_tensor, _) in enumerate(tqdm(dataset)):
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            hidden_states = model.get_intermediate_layers(img_tensor, n=24) 


        grp = h5f.create_group(f"img_{i:05d}")
        for layer_id, layer in enumerate(hidden_states):
            feature = layer.reshape(16, 16, 1024).cpu().numpy()
            grp.create_dataset(f"layer_{layer_id}", data=feature, compression="gzip")

print(f"✅ Done. Features saved to: {hdf5_path}")
