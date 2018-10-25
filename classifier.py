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

from dca_ersvm import ERSVM as DERSVM
from h_ersvm import HeuristicERSVM as HERSVM

def get_classifier(argv, hparam, class_weight):
    """ return classifier with selected arguments """

    if argv.classif == "linear_svm":
        return LinearSVC(C=hparam, class_weight=class_weight,
                         loss="hinge", max_iter=500000)

    if argv.classif == "logi_reg":
        return LogisticRegression(C=hparam, class_weight=class_weight)

    if argv.classif == "random_forest":
        return RandomForestClassifier(class_weight=class_weight, max_depth=hparam)

    if argv.classif == "heuristic_ERSVM":
        assert argv.nb_features > 0, "Number of features required for Robust SVM"
        return HERSVM(class_weight=class_weight, initial_weight=np.random.rand(argv.nb_features))

    if argv.classif == "DCA_ERSVM":
        assert argv.nb_features > 0, "Number of features required for Robust SVM"
        return DERSVM(class_weight=class_weight, initial_w=np.random.rand(argv.nb_features))

    match_svc = re.match("SVC_([A-Za-z]+)", argv.classif)
    if match_svc:
        return SVC(C=hparam, class_weight=class_weight, kernel=match_svc.group(1), max_iter=500000)

    log.error("Unknown classifier %s.", argv.classif)
    sys.exit(0)

def get_confusion(data, nb_class, subset, classif):
    """ Get confusion matrix from classifer """

    conf_mat = np.zeros((nb_class, nb_class))

    if nb_class > 2:
        try:
            if isinstance(classif, list):
                outs = np.zeros((data[subset]["exemples"].shape[0], nb_class))
                for class_i in range(nb_class):
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
