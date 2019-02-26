"""
    Manage Kernel use
"""

import numpy as np

def kernelize(dataset, kernel):
    """ apply kernel to dataset """

    kernel_data = {"nb_class": dataset["nb_class"]}

    nb_folds = len(dataset)-1

    train_size = 0
    for fold_i in range(nb_folds):
        train_size += dataset["fold%d"%fold_i]["train"]["exemples"].shape[0]

    for fold_i in range(nb_folds):
        kernel_data["fold%d"%fold_i] = {}
        k_fold = kernel_data["fold%d"%fold_i]

        fold = dataset["fold%d"%fold_i]

        for subset in ["train", "valid", "test"]:
            k_fold[subset] = {"exemples": np.empty((fold[subset]["exemples"].shape[0], train_size)),
                              "labels":fold[subset]["labels"]}

            k_examples = k_fold[subset]["exemples"]

            slice_i = 0
            for loop_fold_i in range(nb_folds):
                train_examples = dataset["fold%d"%loop_fold_i]["train"]["exemples"]

                k_examples[slice_i:slice_i+train_examples.shape[0]] = kernel_function(fold[subset]["exemples"],
                                                                                      train_examples,
                                                                                      kernel)
                slice_i += train_examples.shape[0]

    return kernel_data

def kernel_function(data, train_data, kernel):
    """ apply given kernel on a subset """

    nb_feat = data.shape[1]

    if kernel["type"] == "rbf":
        return np.exp(-kernel["gamma"]*np.linalg.norm(data.reshape(-1, 1, nb_feat)
                                                      -train_data.reshape(1, -1, nb_feat),
                                                      axis=2))

    raise ValueError("Kernel %s is not implemented yet."%kernel["type"])

def get_kernel_grid(argv):
    """ return kernel tuning loop values """

    if argv.kernel == "rbf":
        return [{"type":"rbf", "gamma": gamma, "label":"rbf_g%s"%(str(gamma).replace(".", "d"))}
                for gamma in argv.gamma_grid]

    raise ValueError("Kernel %s is not implemented yet."%argv.kernel)
