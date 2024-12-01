import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from nltk.tokenize import word_tokenize

class FlickrDataset(Dataset):
    def __init__(self, image_dir, caption_file, vocab, transform=None, max_length=50):
        self.image_dir = image_dir
        self.vocab = vocab
        self.max_length = max_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.image_captions = []
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    image_name = parts[0].split('#')[0]
                    caption = parts[1]
                    if os.path.exists(os.path.join(image_dir, image_name)):
                        self.image_captions.append((image_name, caption))

    def __getitem__(self, idx):
        image_name, caption = self.image_captions[idx]
        
        # 处理图像
        image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
        image = self.transform(image)
        
        # 处理文本
        tokens = ['<START>'] + word_tokenize(caption.lower()) + ['<END>']
        tokens = tokens[:self.max_length]
        
        # 填充到固定长度
        while len(tokens) < self.max_length:
            tokens.append('<PAD>')
            
        # 转换为索引
        caption = [self.vocab.word2idx.get(token, self.vocab.word2idx['<UNK>']) 
                  for token in tokens]
        
        return image, torch.tensor(caption)

    def __len__(self):
        return len(self.image_captions) 