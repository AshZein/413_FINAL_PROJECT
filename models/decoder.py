import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(Decoder, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        # 将图像特征扩展并与每个时间步的词嵌入连接
        features = features.unsqueeze(1).repeat(1, embeddings.size(1), 1)
        lstm_input = torch.cat((features, embeddings), dim=2)
        
        lstm_out, _ = self.lstm(lstm_input)
        outputs = self.linear(lstm_out)
        
        return outputs 