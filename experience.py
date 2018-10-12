"""
    Manage FIAFMO experiences
"""


import os
import math
import sys
import pickle
import argparse
import warnings
import logging as log
from time import localtime, strftime, time

import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC as SVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LogiReg

from . import cone
from . import baselines
from . import parambath

def experience(argv):
    results_path += "/"+strftime("%d%m%Y_%H%M", localtime())+"_"+algo

    try:
        os.makedirs(results_path)
    except FileExistsError:
        # problem when several fold are launched at the same time
        pass


    #load dataset
    #get algo params
    #save 1 log/fold
    #option to avoid keep predictions


if __name__ == "__main__":
    log.basicConfig(stream=sys.stdout, level=log.DEBUG)
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument("--dataset", help="Dataset file", type=str, required=True)
    PARSER.add_argument("--C", help="Tested C, -1 to use {2^-6, 2^-5, ..., 2^6}", type=float, required=True)
    PARSER.add_argument("--log_dir", help="Logs directory", type=str, required=True)
    PARSER.add_argument("--nb_cones", help="Maximum number of cones", type=int, required=True)
    PARSER.add_argument("--beta", help="Beta of F-Measure", type=float, required=True)
    PARSER.add_argument("--fold", help="Tested fold, -1 to use [0, 1, ..., 5]", type=int, required=True)
    PARSER.add_argument("--classif", help="Classifier (logireg|liblinear|rbf|randforest)", type=str,
                        required=True)
    PARSER.add_argument("--algo", help="",
                        type=str, required=True)
    PARSER.add_argument("--max_iter", help="maximum number of training iteration",
                        type=int, required=True)
    PARSER.add_argument("--intercept", help="Fit the intercept of classifier",
                        type=float, required=True)
    PARSER.add_argument("--nb_class", help="Number of classes", type=int, required=True)
    PARSER.add_argument("--cone_subset", type=str, required=True,
                        help="subset of dataset used to draw cones (train|valid|test)")
    PARSER.add_argument("--strategy", type=str, required=True,
                        help="Strategy used to find next cone (middle|left)")
    PARSER.add_argument("--tmin", help="smallest t in grid", type=float, required=True)
    PARSER.add_argument("--tmax", help="highest t in grid", type=float, required=True)

    PARSER.add_argument("--save_states", action='store_true', help="Save all intermediate states")
    PARSER.add_argument("--save_models", action='store_true', help="Save all intermediate models")
    PARSER.add_argument("--save_predictions", action='store_true', help="Save all predictions")

    PARSER.add_argument("--feat_type", type=str, default="none",
                        help="If dataset of features, give their type (ImageNet|trained)")
    PARSER.add_argument("--feat_norm", type=str, default="none",
                        help="If dataset of features, set a normalization (minmax|std|none)")

    experience(PARSER.parse_args())


