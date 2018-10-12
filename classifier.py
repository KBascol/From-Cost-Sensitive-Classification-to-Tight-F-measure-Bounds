"""
    Manage classifiers
"""


import re
import sys
import logging as log

import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# from .dca_ersvm import ERSVM as DERSVM
# from .h_ersvm import HeuristicERSVM as HERSVM

def get_classifier(opt, class_weight):
    """ return classifier with selected arguments """

    if opt["classif"] == "linear_svm":
        return LinearSVC(C=opt["C"], class_weight=class_weight, loss="hinge",
                         max_iter=opt["max_iter"], fit_intercept=(opt["intercept"] > 0),
                         intercept_scaling=opt["intercept"])

    if opt["classif"] == "logi_reg":
        return LogisticRegression(C=opt["C"], class_weight=class_weight,
                                  fit_intercept=(opt["intercept"] > 0),
                                  intercept_scaling=opt["intercept"])

    if opt["classif"] == "random_forest":
        return RandomForestClassifier(class_weight=class_weight)

    # if opt["classif"] == "heuristic_ERSVM":
    #     assert opt["nb_features"] is not None, "Number of features required for Robust SVM"
    #     return HERSVM(class_weight=class_weight, initial_weight=np.random.rand(opt["nb_features"]))

    # if opt["classif"] == "DCA_ERSVM":
    #     assert opt["nb_features"] is not None, "Number of features required for Robust SVM"
    #     return DERSVM(class_weight=class_weight, initial_w=np.random.rand(opt["nb_features"]))

    match_svc = re.match("SVC_([A-Za-z]+)", opt["classif"])
    if match_svc:
        return SVC(C=opt["C"], class_weight=class_weight, kernel=match_svc.group(1), degree=10,
                   max_iter=opt["max_iter"], tol=10**(-5))

    log.error("Unknown classifier %s.", opt["classif"])
    sys.exit(0)

def get_confusion(data, subset, classif):
    """ Get confusion matrix from classifer """

    conf_mat = np.zeros((data["nb_class"], data["nb_class"]))

    if data["nb_class"] > 2:
        try:
            if isinstance(classif, list):
                outs = np.zeros((data[subset]["exemples"].shape[0], data["nb_class"]))
                for class_i in range(data["nb_class"]):
                    outs[:, class_i] = classif[class_i].decision_function(data[subset]["exemples"])
            else:
                outs = classif.decision_function(data[subset]["exemples"])
        except AttributeError:
            # In case of classifier without decision_function (e.g. random forest)
            outs = classif.predict_proba(data[subset]["exemples"])

        outputs = outs.argmax(axis=1)
    else:
        try:
            outs = classif.decision_function(data[subset]["exemples"])

            outputs = (outs > 0).astype(int)
        except AttributeError:
            # In case of classifier without decision_function (e.g. random forest)
            outs = classif.predict_proba(data[subset]["exemples"])
            outputs = np.argmax(outs, axis=1)
            outs = outs.max(axis=1)

    for pred_i, pred in enumerate(outputs):
        conf_mat[data[subset]["labels"][pred_i], pred] += 1

    return conf_mat, outs
