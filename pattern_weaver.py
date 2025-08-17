import numpy as np

def softmax(values):
    exp = np.exp(values)
    return exp / np.sum(exp, axis=0)

def pattern_weaver(n, crystal_values, dimension):

    probs = []

    for i in range(n):
        scores_i = []
        for j in range(n):
            score_ij = (crystal_values[i] * crystal_values[j]) / np.sqrt(dimension)
            scores_i.append(score_ij)

        prob_ijs = softmax(scores_i)
        weighted_probs_i = np.dot(prob_ijs, crystal_values)
        prob_i = np.sum(weighted_probs_i)
        probs.append(prob_i)

    return np.round(probs,3)



if __name__=="__main__":
    crystals = 5
    values = [4, 2, 7, 1, 9]
    dimension = 1
    print(pattern_weaver(crystals, values, dimension))