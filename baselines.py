"""
    Manage baseline algorithms
"""

import logging as log

import numpy as np

from . import classifier

def run_algo(data, classif_opt, algo_opt):
    """ Use baseline algorithm on given options """

    outputs = {"confusions": {"train": np.zeros((1, data["nb_class"], data["nb_class"])),
                              "valid": np.zeros((1, data["nb_class"], data["nb_class"])),
                              "test": np.zeros((1, data["nb_class"], data["nb_class"]))},
               "predictions": {"train": np.zeros((1, data["train"].shape[0], data["nb_class"])),
                               "valid": np.zeros((1, data["valid"].shape[0], data["nb_class"])),
                               "test": np.zeros((1, data["test"].shape[0], data["nb_class"]))},
               "t_values": [-1]}

    conf_mats = outputs["confusions"]
    preds = outputs["predictions"]

    if algo_opt["I.R."]:
        log.info("Baseline I.R %s", str(classif_opt))

        class_w = {class_i:1-(data["train"]["labels"] == class_i).sum()/data["train"]["labels"].shape[0]
                   for class_i in range(data["nb_class"])}
    else:
        log.info("Baseline %s", str(classif_opt))

        class_w = {class_i:1.0 for class_i in range(data["nb_class"])}

    classif = classifier.get_classifier(classif_opt, class_w)

    classif.fit(data["train"]["exemples"], data["train"]["labels"])

    for subset in ["train", "valid", "test"]:
        out_iter = classifier.get_confusion(data, subset, classif)
        conf_mats[subset][0], preds[subset][0] = out_iter

    return outputs
