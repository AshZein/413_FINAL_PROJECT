import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

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
                captions.append(comment)
    return ' '.join(captions)

def create_word_cloud(text):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stop_words,
        collocations=False
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Path to the text file containing captions
file_path = '../flickr30k/captions.txt'
captions_text = load_captions(file_path)
create_word_cloud(captions_text)
