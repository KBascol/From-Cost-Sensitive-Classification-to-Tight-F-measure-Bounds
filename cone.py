"""
    Manage CONE algorithm
"""

import logging as log
from time import time

import numpy as np
from scipy.optimize import fmin_cobyla

from . import utils
from . import classifier
from .manage_state import state_to_png


def run_algo(data, classif_opt, algo_opt):
    """ Use CONE algorithm on given options """

    timer = time()

    outputs = {"confusions": {"train": np.zeros((algo_opt["max_step"], data["nb_class"], data["nb_class"])),
                              "valid": np.zeros((algo_opt["max_step"], data["nb_class"], data["nb_class"])),
                              "test": np.zeros((algo_opt["max_step"], data["nb_class"], data["nb_class"]))},
               "predictions": {"train": np.zeros((algo_opt["max_step"], data["train"].shape[0], data["nb_class"])),
                               "valid": np.zeros((algo_opt["max_step"], data["valid"].shape[0], data["nb_class"])),
                               "test": np.zeros((algo_opt["max_step"], data["test"].shape[0], data["nb_class"]))},
               "t_values": np.full(algo_opt["max_step"], -1)}

    conf_mats = outputs["confusions"]
    preds = outputs["predictions"]

    log.info("CONE %s", str(classif_opt))

    best = [0, 0, None]
    next_t_val = (algo_opt["tmin"]+algo_opt["tmax"])/2
    max_fm = 1

    step = 0

    state = np.ones((10000, 10000))

    log.info("\t\tinitialization time: %f", time()-timer)

    while max_fm >= best[0] and step < algo_opt["max_step"]:
        timer = time()

        log.debug("compute weights...")

        outputs["t_values"][step] = next_t_val
        class_w = compute_weights(outputs["t_values"][step], data["nb_class"], algo_opt["beta"])

        log.debug("initialize classifier...")

        classif = classifier.get_classifier(classif_opt, class_w)

        log.debug("fit classifier...")
        timer_train = time()

        classif.fit(data["train"]["exemples"], data["train"]["labels"])

        log.info("\t\t\ttrain time: %f", time()-timer_train)

        log.debug("test classifier...")
        timer_test = time()

        for subset in ["train", "valid", "test"]:
            out_iter = classifier.get_confusion(data, subset, classif)
            conf_mats[subset][step], preds[subset][step] = out_iter

        log.info("\t\t\ttest time: %f", time()-timer_test)

        timer_cone = time()
        log.debug("select next cone...")

        curr_fm = add_cone(state, conf_mats["train"][step], outputs["t_values"][step], algo_opt)

        log.info("\t\t\tdrawing cone time: %f", time()-timer_cone)

        if curr_fm > best[0]:
            best = [curr_fm, outputs["t_values"][step], classif]

        timer_fm = time()
        next_t_val, max_fm = get_best_fm_available(state, algo_opt["strategy"],
                                                   t_vals=outputs["t_values"][:step+1])
        log.info("\t\t\tfind next cone time: %f", time()-timer_fm)
        log.info("t: %.5f fm: %.5f next t: %.5f max fm: %.5f",
                 outputs["t_values"][step], curr_fm, next_t_val, max_fm)

        if algo_opt["save_states"]:
            state_to_png(state, algo_opt["save_states"], step)

        step += 1

        log.info("\t\tstep %d time: %f", step, time()-timer)

    return outputs

def get_best_fm_available(state, algo_opt, t_vals=None):
    """ search for the next CONE step """

    # search position of next best fmeasure in state
    fm_pos = np.where(state.sum(axis=1) > -100000000)[0]
    if fm_pos.shape[0] == 0:
        log.error("no more cone possible....")
        return -1, -1
    else:
        fm_pos = fm_pos[0]

    # search position of t value of next best fmeasure
    t_pos = np.where(state[fm_pos] == 1)[0]

    if t_pos.shape[0] == 0:
        # if no position, use 0

        t_val = utils.linspace_value(algo_opt["tmin"], algo_opt["tmax"], 10000, 0)
    if t_pos.shape[0] == 1 or algo_opt["strategy"] == "left":
        # if one position or stategy i to take first t from the left

        t_val = utils.linspace_value(algo_opt["tmin"], algo_opt["tmax"], 10000, t_pos[0])
    elif algo_opt["strategy"] == "right":
        # if  stategy i to take first t from the right

        t_val = utils.linspace_value(algo_opt["tmin"], algo_opt["tmax"], 10000, t_pos[-1])
    else:
        # search for the highest longest allowed zone

        first_line = state[fm_pos]

        max_end = max_so_far = first_line[0]
        max_end_i = 0

        for point_i, point in enumerate(first_line[1:]):
            max_end = max(point, max_end + point)

            if max_so_far < max_end:
                max_end_i = point_i+1

            max_so_far = max(max_so_far, max_end)
        t_val = utils.linspace_value(algo_opt["tmin"], algo_opt["tmax"], 10000,
                                     int(max_end_i-max_so_far//2))

    if t_vals is not None and algo_opt["strategy"] == "middle_prev":
        # if strategy is to take the middle of the nearest higher and lower previous t

        prevt_l = t_vals[t_vals < t_val]
        if prevt_l.shape[0] == 0:
            prevt_l = 0
        else:
            prevt_l = np.max(prevt_l)

        prevt_r = t_vals[t_vals > t_val]
        if prevt_r.shape[0] == 0:
            prevt_r = 1
        else:
            prevt_r = np.min(prevt_r)

        t_val = (prevt_l+prevt_r)/2

    return t_val, utils.linspace_value(1, 0, 10000, fm_pos)

def add_cone(state, conf_mat, t_val, algo_opt):
    """ add CONE step on state """

    x_cone = np.linspace(algo_opt["tmin"], algo_opt["tmax"], 10000)
    y_cone = np.linspace(1, 0, 10000)

    curr_fm = utils.micro_fmeasure(conf_mat, algo_opt["beta"])

    if conf_mat.shape[0] == 2:
        slopel, sloper = bin_slope(conf_mat, algo_opt["beta"])
    else:
        slopel, sloper = mclass_slope(conf_mat, algo_opt["beta"])

    offsetl = curr_fm-slopel*t_val
    offsetr = curr_fm-sloper*t_val

    timer = time()
    grid = np.meshgrid(x_cone, y_cone)
    sup_linel = grid[1]-slopel*grid[0]-offsetl >= 0
    sup_liner = grid[1]-sloper*grid[0]-offsetr >= 0
    log.info("\t\t\t\ttimer meshgrid: %f", time()-timer)

    timer = time()
    state[np.logical_and(sup_linel, sup_liner)] = -10000
    log.info("\t\t\t\ttimer draw: %f", time()-timer)

    return curr_fm, state

def bin_slope(conf_mat, beta=1):
    """ get CONE slope from confusion matrix for binary problem """

    nb_pos = conf_mat[1].sum()
    nb_neg = conf_mat[0].sum()
    nb_fp = conf_mat[0, 1]
    nb_fn = conf_mat[1, 0]

    a21 = nb_neg - nb_fp
    a22 = nb_fn*(beta*beta*nb_pos + nb_fp)/(nb_pos - nb_fn + 10**(-9))
    m_left = nb_fp + min(a21, a22)

    m_right = -nb_fn - nb_fp*(nb_pos - nb_fn)/(beta*beta*nb_pos + nb_fp)

    inv_phi = (1+beta**2)*nb_pos - nb_fn + nb_fp

    diff_e = nb_fp - nb_fn

    return ((diff_e-m_left)/inv_phi, (diff_e-m_right)/inv_phi)

def mclass_slope(conf_mat, beta=1):
    """ compute slopes for two lines version of cone"""

    nb_pos = conf_mat[1:].sum()
    nb_fp = conf_mat[0, 1:].sum()
    nb_fn = conf_mat[1:].sum()-np.diagonal(conf_mat[1:, 1:]).sum()

    m_left = mclass_m(conf_mat, False, beta)
    m_right = mclass_m(conf_mat, True, beta)

    inv_phi = (1+beta**2)*nb_pos - nb_fn + nb_fp

    diff_e = nb_fp - nb_fn

    return ((diff_e-m_left)/inv_phi, (diff_e-m_right)/inv_phi)

def mclass_m(conf_mat, mmin=False, beta=1):
    """ get M for twoline mclass setting Mmin if mmin = True else Mmax """

    nb_class = conf_mat.shape[0]
    e_vect = np.array([conf_mat[class_i, np.arange(nb_class) != class_i].sum()
                       for class_i in range(nb_class)])

    if mmin:
        alpha = lambda a: a[0]-np.sum(a[1:])
        init_guess = np.zeros(nb_class)
        init_guess[1:nb_class] = conf_mat[1:nb_class].sum(axis=1)
    else:
        alpha = lambda a: np.sum(a[1:])-a[0]
        init_guess = np.zeros(nb_class)
        init_guess[0] = conf_mat[0].sum()

    all_pos = conf_mat[1:].sum()
    c1_num = beta**2*all_pos+e_vect[0]

    constrains = [] # >= 0 constants
    constrains.append(lambda a: -a[1:].sum()*c1_num/(all_pos-a[1:].sum()+10**(-9))-a[0])

    for class_i in range(nb_class):
        constrains.append(lambda a: a[class_i]-e_vect[class_i])
        constrains.append(lambda a: conf_mat[class_i, class_i]-a[class_i])

    opti_alpha = fmin_cobyla(alpha, init_guess, constrains, maxfun=10000)

    return opti_alpha[0]-opti_alpha[1:].sum()+e_vect[0]-e_vect[1:].sum()

def compute_weights(t_value, nb_class, beta=1.0):
    """ Compute class weights """

    weights = {class_i: (1+beta**2-t_value) for class_i in range(1, nb_class)}
    weights[0] = t_value

    return weights
