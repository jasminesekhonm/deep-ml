import numpy as np 

def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    actual = np.array(actual)
    predicted = np.array(predicted)
    n_samples = len(actual)
    
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    
    confusion_matrix = [[tp, fn], [fp, tn]]

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator > 0 else 0

    accuracy = (tp + tn) / n_samples
    
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    negativePredictive = safe_divide(tn, tn + fn)

    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
