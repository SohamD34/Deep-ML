
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            if actual[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if actual[i] == 1:
                fn += 1
            else:
                fp += 1

    accuracy = (tp + tn) / len(actual)

    confusion_matrix = [[tp, fn], [fp, tn]]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*precision*recall / (precision + recall)

    specificity = tn / (tn + fp)

    negativePredictive = tn / (tn + fn)
    
    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)



actual = [1, 0, 1, 0, 1]
predicted = [1, 0, 0, 1, 1]
print(performance_metrics(actual, predicted))