import numpy as np
from collections import Counter

def bm25_tf(term, document, k1):
        tf = document.count(term)
        return (tf * (k1 + 1)) / (tf + k1) if tf > 0 else 0

def bm25_idf(corpus, term):
        N = len(corpus)
        df = sum(1 for doc in corpus if Counter(doc)[term] > 0)
        return np.log((N + 1) / (df + 1))

def avg_doc_len(corpus):
        total_len = sum(len(doc) for doc in corpus)
        return total_len / len(corpus) if corpus else 0

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
        scores = []
        avg_dl = avg_doc_len(corpus)
        for doc in corpus:
                score = 0
                dl = len(doc)
                for term in query:
                        idf = bm25_idf(corpus, term)
                        tf = doc.count(term)
                        norm = 1 - b + b * (dl / avg_dl)
                        tf_bm25 = (tf * (k1 + 1)) / (tf + k1) if tf > 0 else 0
                        score += idf * tf_bm25 / norm
                scores.append(score)
        return np.round(scores, 3)

corpus = [['the', 'cat', 'sat'], ['the', 'dog', 'ran'], ['the', 'bird', 'flew']]
query = ['the', 'cat']
print(calculate_bm25_scores(corpus, query))
# Expected output - [0.693, 0., 0.]


print(calculate_bm25_scores([['term'] * 10, ['the'] * 2], ['term'], k1=1.0))
