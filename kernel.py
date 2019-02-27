"""
    Manage Kernel use
"""

import numpy as np

def kernelize(fold, kernel):
    """ apply kernel to dataset """

    train_examples = fold["train"]["exemples"]

    k_fold = {}

    for subset in ["train", "valid", "test"]:
        k_fold[subset] = {"exemples": np.empty((fold[subset]["exemples"].shape[0],
                                                train_examples.shape[0])),
                          "labels":fold[subset]["labels"]}

        k_fold[subset]["exemples"] = kernel_func(fold[subset]["exemples"], train_examples, kernel)

    return k_fold

def kernel_func(data, train_data, kernel):
    """ apply given kernel on a subset """

    nb_feat = data.shape[1]

    if kernel["type"] == "precomputed_rbf":
        return np.exp(-kernel["gamma"]*np.linalg.norm(data.reshape(-1, 1, nb_feat)
                                                      -train_data.reshape(1, -1, nb_feat),
                                                      axis=2))

    raise ValueError("Kernel %s is not implemented yet."%kernel["type"])

def get_kernel_grid(argv):
    """ return kernel tuning loop values """

    return [{"type": argv.kernel, "gamma": gamma, "label":"rbf_g%s"%(str(gamma).replace(".", "d"))}
            for gamma in argv.gamma_grid]
