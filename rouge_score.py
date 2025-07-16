def rouge_1_score(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 score between reference and candidate texts.
    
    Returns a dictionary with precision, recall, and f1.
    """
    tokens1 = reference.split(' ')
    tokens2 = candidate.split(' ')

    overlapping_words = set([i for i in tokens1 if i in tokens2])
    overlap_freq = {}
    overlap_count = 0

    for i in overlapping_words:
        count = min(tokens1.count(i), tokens2.count(i))
        overlap_freq[i] = count
        overlap_count += count
    
    precision = overlap_count / len(tokens1)
    recall = overlap_count / len(tokens2)
    f1 = 2*precision*recall / (precision + recall)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    


print(rouge_1_score('the cat sat on the mat', 'the cat is on the mat'))
# {'precision': 0.8333333333333334, 'recall': 0.8333333333333334, 'f1': 0.8333333333333334}