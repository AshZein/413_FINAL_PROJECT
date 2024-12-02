import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class ImageCaptioner(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(ImageCaptioner, self).__init__()
        
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, vocab, max_length=20):
        batch_size = image.size()[0]
        features = self.encoder(image)
        
        # 生成描述
        states = None
        inputs = torch.tensor([[vocab.word2idx['<START>']]] * batch_size, device=image.device)
        caption = [[] for l in range(batch_size)]
        
        for _ in range(max_length):
            embeddings = self.decoder.embed(inputs)
            features_repeat = features.unsqueeze(1)
            #print(f"features_repeat {features_repeat.size()} embeddings {embeddings.size()}")
            lstm_input = torch.cat((features_repeat, embeddings), dim=2)
            lstm_out, states = self.decoder.lstm(lstm_input, states)
            outputs = self.decoder.linear(lstm_out.squeeze(1))
            predicted = outputs.argmax(1)
            #print(f"predicted size {predicted.size()} captions size: {len(caption)} batch size: {batch_size}")
            
            for i in range(predicted.size()[0]):
                if predicted[i].item() != vocab.word2idx['<END>']:
                    caption[i].append(predicted[i].item())
                    inputs[i] = predicted[i].unsqueeze(0).unsqueeze(0)
            # if predicted.item() == vocab.word2idx['<END>']:
            #     break
                
            #caption.append(predicted.item())
            # inputs = predicted.unsqueeze(0).unsqueeze(0)
            
        return caption 