from nltk.translate.bleu_score import corpus_bleu

def calculate_bleu(references, predictions):
    bleu4 = corpus_bleu(references, predictions)
    return bleu4
    