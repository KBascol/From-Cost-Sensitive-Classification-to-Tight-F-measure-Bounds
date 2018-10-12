"""
    Manage Parambath et al. (2014) algorithm
"""

import logging as log

import numpy as np

from . import classifier

def run_algo(data, classif_opt, algo_opt):
    """ Use Parambath et al. (2014) algorithm on given options """

    t_values, step = np.linspace(algo_opt["tmin"], algo_opt["tmax"], algo_opt["grid_size"],
                                 endpoint=False, retstep=True)
    t_values = t_values + step/2

    outputs = {"confusions": {"train": np.zeros((algo_opt["grid_size"], data["nb_class"], data["nb_class"])),
                              "valid": np.zeros((algo_opt["grid_size"], data["nb_class"], data["nb_class"])),
                              "test": np.zeros((algo_opt["grid_size"], data["nb_class"], data["nb_class"]))},
               "predictions": {"train": np.zeros((algo_opt["grid_size"], data["train"].shape[0], data["nb_class"])),
                               "valid": np.zeros((algo_opt["grid_size"], data["valid"].shape[0], data["nb_class"])),
                               "test": np.zeros((algo_opt["grid_size"], data["test"].shape[0], data["nb_class"]))},
               "t_values": t_values}

    conf_mats = outputs["confusions"]
    preds = outputs["predictions"]

    log.info("Parambath %s", str(classif_opt))

    for t_val in t_values:
        class_w = compute_weights(t_val, data["nb_class"], algo_opt["beta"])

        classif = classifier.get_classifier(classif_opt, class_w)

        classif.fit(data["train"]["exemples"], data["train"]["labels"])

        for subset in ["train", "valid", "test"]:
            out_iter = classifier.get_confusion(data, subset, classif)
            conf_mats[subset][step], preds[subset][step] = out_iter

    return outputs

def compute_weights(t_value, nb_class, beta=1.0):
    """ Compute class weights """

    weights = {class_i: (1+beta**2-t_value) for class_i in range(1, nb_class)}
    weights[0] = t_value

    return weights
