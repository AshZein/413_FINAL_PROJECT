import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import gensim.downloader as api
import nltk

# Ensure you have the NLTK stopwords
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load pre-trained word vectors
word_vectors = api.load("glove-wiki-gigaword-100")  # Example using GloVe vectors

def filter_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def plot_word_embeddings(word_list, num_words=200):
    words = [word for word in word_list if word in word_vectors]
    embeddings = [word_vectors[word] for word in words][:num_words]

    # Reduce dimensions with PCA and TSNE
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(pca_result)

    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='blue', edgecolors='k')
    for i, word in enumerate(words[:num_words]):
        plt.annotate(word, xy=(tsne_result[i, 0], tsne_result[i, 1]), fontsize=9)
    plt.title('TSNE plot of Word Embeddings')
    plt.show()
    
def complete_caption_text(f_path):
    whole_captions = ""
    with open(f_path) as f:
        for line in f:
            caption = line.split(",")[-1]
            whole_captions += caption
    return whole_captions

f_path = '../flickr30k/captions.txt'
# Example usage
captions_text = complete_caption_text(f_path)
filtered_words = filter_stopwords(captions_text)
unique_words = list(set(filtered_words))
plot_word_embeddings(unique_words)
