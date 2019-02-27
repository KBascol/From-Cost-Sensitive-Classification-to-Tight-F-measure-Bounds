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

    if kernel["type"] == "precomputed_rbf":
        nb_feat = data.shape[1]

        train_size = train_data.shape[0]

        dist = np.empty(data.shape[0], train_size)

        remaining = train_size%10000
        nb_iter = train_size//10000

        data = data.reshape(-1, 1, nb_feat)
        train_data = train_data.reshape(1, -1, nb_feat)

        for iter_i in range(nb_iter):
            dist[:, iter_i*10000:(iter_i+1)*10000] = ((data-train_data[:, iter_i*10000:(iter_i+1)*10000])**2).sum(axis=2)

        if remaining:
            dist[nb_iter*10000:] = ((data-train_data[:, nb_iter*10000:])**2).sum(axis=2)

        return np.exp(-kernel["gamma"]*dist)

    raise ValueError("Kernel %s is not implemented yet."%kernel["type"])

def get_kernel_grid(argv):
    """ return kernel tuning loop values """

    return [{"type": argv.kernel, "gamma": gamma, "label":"%s_g%s"%(argv.kernel,
                                                                    str(gamma).replace(".", "d"))}
            for gamma in argv.gamma_grid]
