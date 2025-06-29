import torch
from transformers import AutoFeatureExtractor, ViTModel


#加载预训练的DINO，提取imagenet的特征图
class DINOExtractor:
    def __init__(self, device=torch.device("cuda")):
        self.extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vitl14")
        self.model = ViTModel.from_pretrained("facebook/dino-vitl14", output_hidden_states=True).to(device)
        self.model.eval()
        self.devcie = device

    def __call__(self, pil_image):
        inputs = self.extractor(images = pil_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        hidden_states = outputs.hiddden_states 

        features = [ ]
        for layer in hidden_states:
            patch_tokens = layer[:, 1:, :]
            features.append(patch_tokens) 
        return features  # 25层特征图，每一层都是[1, 196, D]