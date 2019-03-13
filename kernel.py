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
        # limitate the size of the matrices during kernel computation...
        max_size = 10000

        dist = np.empty((data.shape[0], train_data.shape[0]), dtype=np.float32)

        train_remaining = train_data.shape[0]%max_size
        train_nb_iter = train_data.shape[0]//max_size

        data_nb_iter = data.shape[0]//max_size

        data = data.reshape(-1, 1, data.shape[1])
        train_data = train_data.reshape(1, -1, train_data.shape[1])

        for data_iter_i in range(data_nb_iter):
            data_start = data_iter_i*max_size
            data_end = data_start+max_size

            for train_iter_i in range(train_nb_iter):
                train_start = train_iter_i*max_size
                train_end = train_start+max_size

                dist[data_start:data_end, train_start:train_end] = ((data[data_start:data_end]
                                                                     -train_data[:, train_start:train_end])**2).sum(axis=2)

            if train_remaining:
                dist[data_start:data_end, train_nb_iter*max_size:] = ((data[data_start:data_end]
                                                                       -train_data[:, train_nb_iter*max_size:])**2).sum(axis=2)

        if data.shape[0]%max_size:
            for train_iter_i in range(train_nb_iter):
                train_start = train_iter_i*max_size
                train_end = train_start+max_size

                dist[data_nb_iter*max_size:, train_start:train_end] = ((data[data_nb_iter*max_size:]
                                                                        -train_data[:, train_start:train_end])**2).sum(axis=2)

            if train_remaining:
                dist[data_nb_iter*max_size:, train_nb_iter*max_size:] = ((data[data_nb_iter*max_size:]
                                                                          -train_data[:, train_nb_iter*max_size:])**2).sum(axis=2)

        return np.exp(-kernel["gamma"]*dist)

    raise ValueError("Kernel %s is not implemented yet."%kernel["type"])

def get_kernel_grid(argv):
    """ return kernel tuning loop values """

    return [{"type": argv.kernel, "gamma": gamma, "label":"%s_g%s"%(argv.kernel,
                                                                    str(gamma).replace(".", "d"))}
            for gamma in argv.gamma_grid]
