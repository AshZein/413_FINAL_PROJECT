import torch
from PIL import Image
import argparse
from torchvision import transforms
from models.image_captioner import ImageCaptioner
import matplotlib.pyplot as plt

def load_image(image_path, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def generate_caption(model, image, vocab, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        caption_ids = model.generate_caption(image, vocab)
        
        # 转换为文本
        caption = []
        for idx in caption_ids:
            word = vocab.idx2word[idx]
            if word == '<END>':
                break
            if word not in ['<START>', '<PAD>']:
                caption.append(word)
    
    return ' '.join(caption)

def visualize_result(image_path, caption):
    # 显示图像和生成的描述
    plt.figure(figsize=(10, 5))
    
    # 显示图像
    plt.subplot(1, 2, 1)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Input Image')
    
    # 显示描述
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, caption, 
             horizontalalignment='center',
             verticalalignment='center',
             wrap=True)
    plt.axis('off')
    plt.title('Generated Caption')
    
    plt.tight_layout()
    plt.show()

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载检查点
    checkpoint = torch.load(args.model_path, map_location=device)
    vocab = checkpoint['vocab']
    
    # 初始化模型
    model = ImageCaptioner(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab)
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载和预处理图像
    image = load_image(args.image_path)
    
    # 生成描述
    caption = generate_caption(model, image, vocab, device)
    print(f"\nGenerated caption: {caption}")
    
    # 可视化结果
    if not args.no_plot:
        visualize_result(args.image_path, caption)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Image Caption')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='Word embedding size')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='LSTM hidden size')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable result visualization')
    
    args = parser.parse_args()
    main(args) 