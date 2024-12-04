import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
from data.dataset import FlickrDataset
from data.vocabulary import Vocabulary
from models.image_captioner import ImageCaptioner
from utils.metrics import calculate_bleu
import argparse
import time
from datetime import datetime

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for i, (images, captions) in enumerate(progress_bar):
        images = images.to(device)
        captions = captions.to(device)
        
        # 准备输入和目标
        targets = captions[:, 1:]  # 移除START token
        inputs = captions[:, :-1]  # 移除END token
        
        # 前向传播
        outputs = model(images, inputs)
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), 
                        targets.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})
        
        # 记录到TensorBoard
        step = epoch * len(dataloader) + i
        writer.add_scalar('Training/BatchLoss', loss.item(), step)
    
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Training/EpochLoss', avg_loss, epoch)
    return avg_loss

def validate(model, dataloader, criterion, vocab, device):
    model.eval()
    all_predictions = []
    all_references = []
    total_val_loss = 0.0
    #count = 0 # Used for counting iterations for printing out predicted captions after
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            #print(f"number of captions: {len(captions)}")
            # 生成描述
            predicted_ids = model.generate_caption(images, vocab)
            
            # 转换为文本
            predicted_caption = [[] for i in range(len(predicted_ids))]
            for i in range(len(predicted_ids)):
                for idx in predicted_ids[i]:
                    if idx not in [vocab.word2idx['<START>'], vocab.word2idx['<END>'], vocab.word2idx['<PAD>']]:
                        predicted_caption[i].append(vocab.idx2word[idx])
                        
            reference_caption = [[vocab.idx2word[idx.item()] for idx in caption[1:-1]] for caption in captions]
            
            outputs = model(images, captions)
            val_loss = criterion(outputs, captions)
            total_val_loss += val_loss.item()
            
            all_predictions.extend([predicted_caption])
            all_references.extend([reference_caption])
    #print(f"len of all_ref: {len(all_references)} len of all_preds: {len(all_predictions)}")
    # 计算BLEU分数
    bleu4 = 0.0
    for i in range(len(all_predictions)):
        bleu4 += calculate_bleu(all_references[i], all_predictions[i])
    print(f"Total Validation Loss:{total_val_loss}")
    return bleu4 / len(all_predictions) # average of all bleu scores across each in batch

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    
    # 加载或创建词汇表
    vocab_path = os.path.join(args.save_dir, 'vocab.pkl')
    if os.path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
        print("Loaded existing vocabulary")
    else:
        vocab = Vocabulary()
        vocab.build_vocab(args.caption_file, min_freq=args.min_word_freq)
        vocab.save(vocab_path)
        print("Created new vocabulary")
    
    # 创建数据集和数据加载器
    dataset = FlickrDataset(args.image_dir, args.caption_file, vocab)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"training set size:{train_size} validation set size: {val_size}")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 初始化模型
    model = ImageCaptioner(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab)
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # 训练循环
    best_bleu = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        print(f"Training Loss: {train_loss:.4f}")
        
        # 验证
        bleu_score = validate(model, val_loader, criterion, vocab, device)
        print(f"BLEU-4 Score: {bleu_score:.4f}")
        
        # 记录到TensorBoard
        writer.add_scalar('Validation/BLEU4', bleu_score, epoch)
        
        # 学习率调整
        scheduler.step(bleu_score)
        
        # 保存最佳模型
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'bleu_score': bleu_score,
                'vocab': vocab,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved new best model with BLEU-4: {bleu_score:.4f}")
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Captioning Model')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--caption_file', type=str, required=True,
                        help='Path to caption file')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save models')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='Word embedding size')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='LSTM hidden size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--min_word_freq', type=int, default=5,
                        help='Minimum word frequency threshold')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    print(f"Started training at: {start_time.strftime('%d-%m %H:%M')}")
    t0 = time.time()
    main(args) 
    t1 = time.time()
    
    elapsed_time = t1 - t0
    hours = elapsed_time // 3600
    mins = (elapsed_time % 3600) //60
    print(f"Time Taken: {hours} hours {mins} minutes")