"""
Handle CONE states
"""

import os
import sys

import numpy as np
import skimage.io
import skimage.transform

from bluepy.utils import fileio


def state_to_png(state, repo, cone_i=None):
    """ get an image from a state """

    if cone_i is None:
        cones = range(5)
    else:
        cones = [cone_i]

    for cone in cones:
        if isinstance(state, str):
            state_mat = np.load(state).item()["states"][cone]
        else:
            state_mat = state

        skimage.io.imsave("%s/state_cone%d.png"%(repo, cone),
                          skimage.transform.resize((state_mat+10000)/10001, (1000, 1000)))

def grid_to_json(steps_path, dataset, c_value=None):
    """ get json files from grid results file """

    results = np.load(steps_path).item()

    if c_value is not None and not isinstance(c_value, list):
        c_value = [c_value]
    elif c_value is None:
        c_value = [2**c_val_i for c_val_i in range(-6, 7)]


    for c_val in c_value:
        conf_mat = results[c_val]["train"]["confs"]

        to_json = {"points":[], "P":conf_mat[0][1].sum(), "N":conf_mat[0][0].sum(), "beta":1.0,
                   "C":c_val, "dataset":dataset}

        for t_val_i, t_val in enumerate(np.arange(0.1, 2, 0.1)):
            to_json["points"].append({"t":t_val,
                                      "fp":conf_mat[t_val_i][0, 1],
                                      "fn":conf_mat[t_val_i][1, 0]})

        fileio.write_json("%s/pts_grid_%s_C%s.json"%(os.path.dirname(steps_path), dataset,
                          str(c_val).replace(".", "")), to_json)

def cones_to_json(steps_path, dataset, c_value=None):
    """ get json files from cone results file """

    if c_value is not None and not isinstance(c_value, list):
        c_value = [c_value]
    elif c_value is None:
        c_value = [2**c_val_i for c_val_i in range(-6, 7)]

    for c_val in c_value:
        steps = np.load(steps_path%str(c_val).replace(".", "")).item()

        to_json = {"points":[], "P":steps["conf_train"][0][1].sum(),
                   "N":steps["conf_train"][0][0].sum(), "beta":1.0,
                   "C":c_val, "dataset":dataset}

        for t_val, conf_mat in zip(steps["t_values"], steps["conf_train"]):
            if steps["conf_train"].sum() != 0:
                to_json["points"].append({"t":t_val,
                                          "fp":conf_mat[0, 1],
                                          "fn":conf_mat[1, 0]})

        fileio.write_json("%s/pts_cone_%s_C%s.json"%(os.path.dirname(steps_path), dataset,
                          str(c_val).replace(".", "")), to_json)


if __name__ == "__main__":
    for state_path in sys.argv[1:]:
        state_to_png(state_path, "/tmp")
