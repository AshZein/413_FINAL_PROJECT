import torch
import torch.nn as nn
from transformers import ViTModel

class Encoder(nn.Module):
    def __init__(self, embed_size=256):
        super(Encoder, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.linear = nn.Linear(768, embed_size)
        
        # 冻结ViT参数
        for param in self.vit.parameters():
            param.requires_grad = False
            
    def forward(self, images):
        features = self.vit(images).last_hidden_state[:, 0, :]  # 使用[CLS]标记的输出
        features = self.linear(features)
        return features 