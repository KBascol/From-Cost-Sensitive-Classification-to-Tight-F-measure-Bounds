"""
    Manage Parambath et al. (2014) algorithm
"""

import numpy as np

import classifier

def run_algo(data, nb_class, hparam, argv):
    """ Use Parambath et al. (2014) algorithm on given options """

    # grid on t: covers all space with half a step at each extremity
    t_values, step = np.linspace(argv.tmin, argv.tmax, argv.max_step,
                                 endpoint=False, retstep=True)
    t_values = t_values + step/2

    outputs = {"confusions": {"train": np.zeros((argv.max_step, nb_class, nb_class), dtype=int),
                              "valid": np.zeros((argv.max_step, nb_class, nb_class), dtype=int),
                              "test": np.zeros((argv.max_step, nb_class, nb_class), dtype=int)},
               "predictions": {"train": np.zeros((argv.max_step, data["train"]["labels"].shape[0], nb_class), dtype=np.float32),
                               "valid": np.zeros((argv.max_step, data["valid"]["labels"].shape[0], nb_class), dtype=np.float32),
                               "test": np.zeros((argv.max_step, data["test"]["labels"].shape[0], nb_class), dtype=np.float32)},
               "t_values": t_values}

    conf_mats = outputs["confusions"]
    preds = outputs["predictions"]

    for t_val_i, t_val in enumerate(t_values):
        class_w = compute_weights(t_val, nb_class, argv.beta)

        classif = classifier.get_classifier(argv, hparam, class_w)

        classif.fit(data["train"]["exemples"], data["train"]["labels"])

        for subset in ["train", "valid", "test"]:
            out_iter = classifier.get_confusion(data, nb_class, subset, classif)

            if nb_class == 2:
                saved_outs = preds[subset][t_val_i][:, 0]
            else:
                saved_outs = preds[subset][t_val_i]

            conf_mats[subset][t_val_i], saved_outs = out_iter

    return outputs

def compute_weights(t_value, nb_class, beta=1.0):
    """ Compute class weights """

    weights = {class_i: (1+beta**2-t_value) for class_i in range(1, nb_class)}
    weights[0] = t_value

    return weights
