import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk import bigrams, trigrams
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_captions(file_path):
    captions = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by commas and take the third part as the comment
            parts = line.strip().split(',')
            if len(parts) >= 3:
                comment = parts[2]
                if comment[-1] == ".":
                    comment = comment[:-1]
                captions.append(comment)
    return captions

def filter_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def plot_bigrams_trigrams(captions, n=20):
    bigram_counts = Counter()
    trigram_counts = Counter()

    for caption in captions:
        tokens = filter_stopwords(caption)
        bigram_counts.update(bigrams(tokens))
        trigram_counts.update(trigrams(tokens))

    common_bigrams = bigram_counts.most_common(n)
    common_trigrams = trigram_counts.most_common(n)

    plt.figure(figsize=(12, 6))

    # Plot bigrams
    plt.subplot(1, 2, 1)
    bigrams_text = [' '.join(pair) for pair, count in common_bigrams]
    bigrams_counts = [count for pair, count in common_bigrams]
    sns.barplot(x=bigrams_counts, y=bigrams_text)
    plt.title('Top Bigrams')

    # Plot trigrams
    plt.subplot(1, 2, 2)
    trigrams_text = [' '.join(trio) for trio, count in common_trigrams]
    trigrams_counts = [count for trio, count in common_trigrams]
    sns.barplot(x=trigrams_counts, y=trigrams_text)
    plt.title('Top Trigrams')

    plt.tight_layout()
    plt.show()

# Path to the text file containing captions
file_path = '../flickr30k/captions.txt'
captions = load_captions(file_path)
plot_bigrams_trigrams(captions)
