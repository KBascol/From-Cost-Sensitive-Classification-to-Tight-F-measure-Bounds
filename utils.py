"""
    tools and measures
"""

import json
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

def comp_fm(conf_mat, beta=1.0, tune_thresh=False, out=None, label=None):
    """ return f-measure and threshold if tuning enable None otherwise """

    if tune_thresh:
        if conf_mat.shape[0] > 2:
            thresholds = np.max(out[:, 1:], axis=1)-out[:, 0]+1
        if out.shape[1] == 2:
            thresholds = out[:, 0]

        thresholds = np.linspace(thresholds.min(), thresholds.max(), min(10000, out.shape[0]))

        fmeas = np.zeros(thresholds.shape[0])

        for fm_i, thres in enumerate(thresholds):
            fmeas[fm_i] = micro_fmeasure(thresh_conf(out, label, thres), beta)

        best_i = np.argmax(fmeas)
        thres = out[best_i]

        if conf_mat.shape[0] > 2:
            thres = np.max(thres[1:])-thres[0]+1
        if out.shape[1] == 2:
            thresholds = thres[0]

        return (fmeas[best_i], thres)
    else:
        return (micro_fmeasure(conf_mat, beta), None)

def thresh_conf(outs, label, threshold):
    """ get confusion matrix given a prediction threshold """

    nb_class = outs.shape[1]

    conf_mat = np.zeros((nb_class, nb_class))

    if nb_class == 2:
        outputs = (outs[:, 0] > threshold).astype(int)
    else:
        outputs = np.argmax(outs + threshold, axis=1)

    for pred_i, pred in enumerate(outputs):
        conf_mat[label[pred_i], pred] += 1

    return conf_mat

def micro_fmeasure(conf_mat, beta=1.0):
    """ Compute micro-Fmeasure for multiclass problems """

    fmeas = (1+beta**2)*(np.diagonal(conf_mat[1:, 1:]).sum())
    fmeas /= (beta**2)*(conf_mat[1:].sum())\
             +np.diagonal(conf_mat[1:, 1:]).sum()\
             +conf_mat[0, 1:].sum()

    return fmeas

def load_json(file_path):
    """ Parse a file content for remove unsupported comments and load as JSON """
    with open(file_path) as json_file:
        return json.load(json_file)
    raise Exception("Impossible to read JSON file " + file_path)

def write_json(file_path, content):
    """ Parse a file content for remove unsupported comments and load as JSON """
    with open(file_path, 'w') as json_file:
        json_file.write(json.dumps(content, indent=4 * ' '))
