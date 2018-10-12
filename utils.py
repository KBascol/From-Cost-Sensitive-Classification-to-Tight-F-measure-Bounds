"""
    tools and measures
"""

import logging as log

import numpy as np

def change_log_file(log_file):
    """ Change the fileHandler of the default logger """

    logger = log.getLogger()
    logger.removeHandler(logger.handlers[0])
    logger.addHandler(log.FileHandler(log_file))

def labels_to_class(label, nb_class):
    """ vector label to value """

    if nb_class == 2:
        return np.array([label[-1] == 1], dtype=int)

    return np.array([np.argmax(label)], dtype=int)

def linspace_value(start, end, nb_values, position):
    """ value of a position in a linear space """

    return ((end - start)/(nb_values-1))*position + start

def accuracies(conf, class_w, nb_class):
    """ compute accuracy and weighted accuracy """

    array_class_w = np.array([class_w[val_i] for val_i in range(nb_class)])

    acc = np.diag(conf).sum()/conf.sum()
    wacc = (np.diag(conf)*array_class_w).sum()/(conf*array_class_w).sum()

    return acc, wacc

def bin_fmeasure(conf_mat, beta=1.0):
    """ Compute f-measure from confusion matrix """

    true_p = conf_mat[1, 1]

    if true_p == 0:
        return 0.0

    tpfp = conf_mat[:, 1].sum()

    if tpfp > 0:
        preci = true_p/tpfp
    else:
        preci = 1

    recall = true_p/(conf_mat[1].sum())

    return (beta**2+1)*preci*recall/(beta**2*preci+recall)

def preci_recall_class(conf_mat, class_i):
    """ Compute precision dans recall from confusion matrix """

    true_p = conf_mat[class_i, class_i]
    tpfp = conf_mat[:, class_i].sum()

    if tpfp > 0:
        precision = true_p/tpfp
    else:
        precision = 1

    recall = true_p/(conf_mat[class_i].sum())

    return (precision, recall)

def macro_fmeasure(conf_mat, beta=1.0):
    """ get fmeasure from confusion matrix for multiclass problem """

    precisions = np.zeros(conf_mat.shape[0])
    recalls = np.zeros(conf_mat.shape[0])

    for class_i in range(conf_mat.shape[0]):
        precisions[class_i], recalls[class_i] = preci_recall_class(conf_mat, class_i)

    mean_recall = recalls.mean()
    mean_preci = precisions.mean()

    return ((1+beta**2)*mean_preci*mean_recall)/(beta**2*mean_preci+mean_recall)

def micro_fmeasure(conf_mat, beta=1.0):
    """ Compute micro-Fmeasure for multiclass problems """

    fmeas = (1+beta**2)*(np.diagonal(conf_mat[1:, 1:]).sum())
    fmeas /= (beta**2)*(conf_mat[1:].sum())\
             +np.diagonal(conf_mat[1:, 1:]).sum()\
             +conf_mat[0, 1:].sum()

    return fmeas
