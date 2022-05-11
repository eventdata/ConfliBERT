import numpy as np


def example_based_accuracy(y_true_, y_pred_):

    # pre-process the inputs
    y_true = []
    y_pred = []
    for true, pred in zip(y_true_, y_pred_):
        if sum(true+pred) > 0:
            y_true.append(true)
            y_pred.append(pred)

    if len(y_true) == 0:
        return float(0)

    # compute true positives using the logical AND operator
    numerator = np.sum(np.logical_and(y_true, y_pred), axis = 1)

    # compute true_positive + false negatives + false positive using the logical OR operator
    denominator = np.sum(np.logical_or(y_true, y_pred), axis = 1)
    instance_accuracy = numerator/denominator

    avg_accuracy = np.mean(instance_accuracy)
    return float(avg_accuracy)




def example_based_recall(y_true_, y_pred_):

    # pre-process the inputs
    y_true = []
    y_pred = []
    for true, pred in zip(y_true_, y_pred_):
        if sum(true) > 0:
            y_true.append(true)
            y_pred.append(pred)

    if len(y_true) == 0:
        return float(0)

    # Compute True Positive 
    recall_num = np.sum(np.logical_and(y_true, y_pred), axis = 1)

    # Total number of actual true labels
    recall_den = np.sum(y_true, axis = 1)

    # recall averaged over all training examples
    avg_recall = np.mean(recall_num/recall_den)

    return float(avg_recall)




def example_based_precision(y_true_, y_pred_):

    # pre-process the inputs
    y_true = []
    y_pred = []
    for true, pred in zip(y_true_, y_pred_):
        if sum(pred) > 0:
            y_true.append(true)
            y_pred.append(pred)

    if len(y_true) == 0:
        return float(0)


    # Compute True Positive 
    prec_num = np.sum(np.logical_and(y_true, y_pred), axis = 1)

    # Total number of actual true labels
    prec_den = np.sum(y_pred, axis = 1)

    # recall averaged over all training examples
    avg_prec = np.mean(prec_num/prec_den)

    return float(avg_prec)



def example_based_f1(y_true_, y_pred_):

    # pre-process the inputs
    y_true = []
    y_pred = []
    for true, pred in zip(y_true_, y_pred_):
        if sum(true+pred) > 0:
            y_true.append(true)
            y_pred.append(pred)

    if len(y_true) == 0:
        return float(0)

    # Compute True Positive 
    f1_num = np.sum(np.logical_and(y_true, y_pred), axis = 1)

    # Total number of actual true labels
    f1_den = np.sum(y_pred, axis = 1)+np.sum(y_true, axis = 1)

    # recall averaged over all training examples
    avg_f1 = np.mean((2*f1_num)/f1_den)

    return float(avg_f1)


