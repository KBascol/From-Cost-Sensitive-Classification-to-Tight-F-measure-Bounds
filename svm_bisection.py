"""
Consistent Multiclass Algorithms for Complex Performance Measures Algorithm 2 p6
"""

import os
import sys
import argparse
import logging as log
from time import localtime, strftime

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression as LogiReg


def predict(dataset, tuned_classif):
    """ Get tuned prediction """

    pred = tuned_classif[0].predict_proba(dataset)

    loss_pred = np.matmul(np.transpose(tuned_classif[1]), np.transpose(pred))

    return np.argmin(loss_pred, axis=0)

def run_bisection(log_dir, dataset_path, nb_class, fold, c_value, beta=1, kappa=1, intercept=1,
                  max_iter=50000):
    """ Tune classifier with bisection algorithm """
    log_dir += "/"+strftime("%d%m%Y_%H%M", localtime())+"_bissection"

    try:
        os.makedirs(log_dir)
    except FileExistsError:
        # problem when several fold are launched at the same time
        pass

    disp_b = ("%.1f"%beta).replace(".", "")

    a_mat = np.zeros((nb_class, nb_class))
    np.fill_diagonal(a_mat, 1+beta**2)
    a_mat[0, 0] = 0

    b_mat = np.full((nb_class, nb_class), 1+beta**2)
    b_mat[0, 0] = 0
    b_mat[0, 1:] = 1
    b_mat[1:, 0] = beta**2

    log.info("load dataset...")

    dataset = np.load(dataset_path)

    if isinstance(dataset, np.ndarray):
        dataset = dataset.item()
    else:
        dataset = dataset['dataset'].item()

    if fold == -1:
        folds = list(range(5))
    else:
        folds = [fold]

    if c_value == -1:
        c_value = [2**exp for exp in range(-6, 7)]
    else:
        c_value = [c_value]

    for fold_i in folds:
        change_log_file(log_dir+"/stdout_f%d.log"%fold_i)

        log.info("##########")
        log.info(" FOLD n°%d ", fold_i)
        log.info("##########")

        data_train = dataset["folds"][fold_i]["train"][0]
        label_train = dataset["folds"][fold_i]["train"][1].astype(int)
        data_valid = dataset["folds"][fold_i]["valid"][0]
        label_valid = dataset["folds"][fold_i]["valid"][1].astype(int)
        data_test = dataset["folds"][fold_i]["test"][0]
        label_test = dataset["folds"][fold_i]["test"][1].astype(int)

        results = {}

        for c_val in c_value:
            log.info("\t C %f", c_val)
            results[c_val] = {"train":{"confs":[], "fmeas":[], "gamma":[]},
                              "valid":{"confs":[], "fmeas":[], "gamma":[]},
                              "test":{"confs":[], "fmeas":[], "gamma":[]}}

            classif = LogiReg(C=c_val, max_iter=max_iter, fit_intercept=intercept > 0,
                              intercept_scaling=intercept)

            log.info("\t fit logistic regression...")
            classif.fit(data_train, label_train)

            borne_inf = 0
            borne_sup = 1
            tuned_classif = [classif, np.ones((nb_class, nb_class)), -1]

            append_res(results[c_val], "train", data_train, tuned_classif, label_train, beta)
            append_res(results[c_val], "valid", data_valid, tuned_classif, label_valid, beta)
            append_res(results[c_val], "test", data_test, tuned_classif, label_test, beta)

            # nb_iter = int(kappa*(label_train.shape[0]+label_valid.shape[0])) # TOO LONG !!!!
            nb_iter = int(kappa)
            for iter_i in range(nb_iter):
                log.info("\t tuning iteration n°%d/%d...", iter_i, nb_iter)

                gamma = (borne_inf+borne_sup)/2

                loss_mat = -(a_mat-gamma*b_mat)
                loss_mat = (loss_mat-loss_mat.min())/(loss_mat.max()-loss_mat.min())

                conf = confusion_matrix(label_valid, predict(data_valid, [classif, loss_mat]))

                results[c_val]["valid"]["confs"].append(conf)
                results[c_val]["valid"]["gamma"].append(gamma)
                nb_tp = np.diagonal(conf[1:, 1:]).sum()
                fmeas = (1+beta**2)*nb_tp/(nb_tp+(beta**2)*conf[1:].sum()+conf[0, 1:].sum())
                results[c_val]["valid"]["fmeas"].append(fmeas)

                phi = (a_mat*conf).sum()/(b_mat*conf).sum()

                if phi >= gamma:
                    borne_inf = gamma
                    tuned_classif = [classif, loss_mat, gamma]
                else:
                    borne_sup = gamma
                log.info("\t gamma = %f; phi = %f", gamma, phi)

            append_res(results[c_val], "train", data_train, tuned_classif, label_train, beta)
            append_res(results[c_val], "valid", data_valid, tuned_classif, label_valid, beta)
            append_res(results[c_val], "test", data_test, tuned_classif, label_test, beta)

        np.save(log_dir+"/results_bissection_b%s_f%d.npy"%(disp_b, fold_i), results)


def change_log_file(log_file):
    """ Change the fileHandler of the default logger """

    logger = log.getLogger()
    logger.removeHandler(logger.handlers[0])
    logger.addHandler(log.FileHandler(log_file))

def append_res(results_c, subset, data, tuned_classif, labels, beta=1):
    """ append current result in dictoonnary """

    preds = predict(data, tuned_classif)

    conf = confusion_matrix(preds, labels)

    results_c[subset]["confs"].append(conf)

    results_c[subset]["gamma"].append(tuned_classif[2])

    nb_tp = np.diagonal(conf[1:, 1:]).sum()
    fmeas = (1+beta**2)*nb_tp/(nb_tp+(beta**2)*conf[1:].sum()+conf[0, 1:].sum())
    results_c[subset]["fmeas"].append(fmeas)

if __name__ == "__main__":
    log.basicConfig(stream=sys.stdout, level=log.DEBUG)
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument("--dataset", help="Dataset file", type=str, required=True)
    PARSER.add_argument("--C", help="Tested C, -1 to do all", type=float, required=True)
    PARSER.add_argument("--log_dir", help="Logs directory", type=str, required=True)
    PARSER.add_argument("--beta", help="Beta of F-Measure", type=float, required=True)
    PARSER.add_argument("--fold", help="Tested fold, -1 to do all", type=int, required=True)
    PARSER.add_argument("--max_iter", help="maximum number of training iteration",
                        type=int, required=True)
    PARSER.add_argument("--intercept", help="Fit the intercept of classifier",
                        type=float, required=True)
    PARSER.add_argument("--nb_class", help="Number of classes", type=int, required=True)
    PARSER.add_argument("--kappa", help="Number of 'epochs'", type=float, required=True)

    ARGP = PARSER.parse_args()

    run_bisection(log_dir=ARGP.log_dir, dataset_path=ARGP.dataset, nb_class=ARGP.nb_class,
                  fold=ARGP.fold, c_value=ARGP.C, beta=ARGP.beta, kappa=ARGP.kappa,
                  max_iter=ARGP.max_iter, intercept=ARGP.intercept)
