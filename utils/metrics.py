from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

def calculate_bleu(references, predictions):
    smoothing_function = SmoothingFunction().method1
    bleu4 = corpus_bleu(references, predictions, weights=(0, 0, 1, 0), smoothing_function=smoothing_function)
    return bleu4
    