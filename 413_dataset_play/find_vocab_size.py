import os

file_path = '../flickr30k/captions.txt'
vocab = {}
with open(file_path) as f:
    for line in f:
        caption = line.split(",")[-1]
        for word in caption.split():
            if word != ".":
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

print(f"vocab size {len(vocab.keys())}")
sorted_vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
print(list(sorted_vocab.keys())[-10:])