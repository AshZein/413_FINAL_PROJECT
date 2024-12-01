import pickle
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
import re
nltk.download('punkt')
from utils.validate_data import validate_dataset
import torch
import torch.nn as nn
from torchvision import models
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import cv2
import os

class Vocabulary:
    def __init__(self):
        self.word2idx = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNK>': 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()
        self.idx = 4

    def build_vocab(self, caption_file, min_freq=5):
        def clean_text(text):
            # 移除标点符号和特殊字符
            text = re.sub(r'[^\w\s]', '', text)
            # 转换为小写
            text = text.lower()
            return text
        
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    caption = line.strip().split('\t')[1]  # Flickr30k格式
                    caption = clean_text(caption)
                    tokens = word_tokenize(caption)
                    self.word_freq.update(tokens)
                except Exception as e:
                    print(f"Error processing line: {line.strip()}\nError: {e}")
        
        # 添加频率超过阈值的词
        for word, freq in self.word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f) 

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        
        # 冻结ResNet参数
        for param in self.resnet.parameters():
            param.requires_grad = False
            
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        return features

def calculate_bleu(references, hypotheses):
    return corpus_bleu(references, hypotheses) * 100

def pad_image(img):
    # 实现图像填充逻辑
    # 返回填充后的图像
    pass

def save_image(img, filename, output_dir):
    padded_img = pad_image(img)
    cv2.imwrite(os.path.join(output_dir, filename), padded_img)