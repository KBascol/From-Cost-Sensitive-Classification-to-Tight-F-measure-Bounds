"""
    Manage CONE algorithm
"""

import math
import logging as log
from time import time

import numpy as np
import skimage.io
import skimage.transform
from scipy.optimize import linprog

import utils
import classifier

def run_algo(data, nb_class, hparam, argv):
    """ Use CONE algorithm on given options """

    overall_timer = time()

    outputs = {"confusions": {"train": np.zeros((argv.max_step, nb_class, nb_class), dtype=int),
                              "valid": np.zeros((argv.max_step, nb_class, nb_class), dtype=int),
                              "test": np.zeros((argv.max_step, nb_class, nb_class), dtype=int)},
               "predictions": {"train": np.zeros((argv.max_step, data["train"]["labels"].shape[0], nb_class), dtype=int),
                               "valid": np.zeros((argv.max_step, data["valid"]["labels"].shape[0], nb_class), dtype=int),
                               "test": np.zeros((argv.max_step, data["test"]["labels"].shape[0], nb_class), dtype=int)},
               "t_values": np.full(argv.max_step, -1, dtype=np.float32)}

    conf_mats = outputs["confusions"]
    preds = outputs["predictions"]

    best_fm = 0
    next_t_val = (argv.tmin+argv.tmax)/2
    max_fm = 1

    step = 0

    if argv.cone_with_state:
        state = np.ones((argv.state_size, argv.state_size))
    else:
        saved_cones = np.zeros((argv.max_step+1, 6))
        # add line y = 1 as "cone" to get early intersections points
        saved_cones[0] = [0, -1, 0, 0, 1, 1]

        poss_next_t = np.empty(0, dtype=np.float32)
        poss_next_fm = np.empty(0, dtype=np.float32)

    while max_fm >= best_fm and step < argv.max_step:
        timer = time()

        log.debug("Compute weights...")

        outputs["t_values"][step] = next_t_val
        class_w = compute_weights(outputs["t_values"][step], nb_class, argv.beta)

        log.debug("Initialize classifier...")

        classif = classifier.get_classifier(argv, hparam, class_w)

        log.debug("Fit classifier...")
        timer_train = time()

        classif.fit(data["train"]["exemples"], data["train"]["labels"])

        log.debug("\ttrain time: %f", time()-timer_train)

        log.debug("Test classifier...")
        timer_test = time()

        for subset in ["train", "valid", "test"]:
            out_iter = classifier.get_confusion(data, nb_class, subset, classif)

            if nb_class == 2:
                conf_mats[subset][step], preds[subset][step][:, 0] = out_iter
            else:
                conf_mats[subset][step], preds[subset][step] = out_iter

        log.debug("\ttest time: %f", time()-timer_test)

        log.debug("Select next cone...")

        timer_cone = time()
        if argv.cone_with_state:
            timer_draw = time()
            curr_fm = add_cone(state, conf_mats["train"][step], outputs["t_values"][step], argv)
            log.debug("\t\tdrawing cone time: %f", time()-timer_draw)

            timer_fm = time()
            next_t_val, max_fm = get_best_fm_available(state, argv, t_vals=outputs["t_values"][:step+1])
            log.debug("\t\tfind next cone time: %f", time()-timer_fm)

            if argv.save_states:
                state_to_png(state, argv, step)
        else:
            saved_cones[step+1] = get_slope(conf_mats["train"][step], outputs["t_values"][step],
                                            "cone", beta=argv.beta)
            curr_fm = saved_cones[step+1][0]
            next_t_val, max_fm, poss_next_t, poss_next_fm = find_next_cone(saved_cones[:step+2],
                                                                           outputs["t_values"][:step+1],
                                                                           poss_next_t, poss_next_fm,
                                                                           tmin=argv.tmin, tmax=argv.tmax)
        log.debug("\tnext cone time: %f", time()-timer_cone)

        if curr_fm > best_fm:
            best_fm = curr_fm

        log.debug("Step %d time: %f t: %.5f fm: %.5f next t: %.5f max fm: %.5f", step, time()-timer,
                  outputs["t_values"][step], curr_fm, next_t_val, max_fm)

        step += 1

    log.debug("TRAIN TIME: %f", time()-overall_timer)

    return outputs

def get_best_fm_available(state, argv, t_vals=None):
    """ search for the next CONE step """

    # search position of next best fmeasure in state
    fm_pos = np.where(state.sum(axis=1) > -argv.state_size**2)[0]
    if fm_pos.shape[0] == 0:
        log.error("no more cone possible....")
        return -1, -1
    else:
        fm_pos = fm_pos[0]

    # search position of t value of next best fmeasure
    t_pos = np.where(state[fm_pos] == 1)[0]

    if t_pos.shape[0] == 0:
        # if no position, use 0

        t_val = utils.linspace_value(argv.tmin, argv.tmax, argv.state_size, 0)
    if t_pos.shape[0] == 1 or argv.strategy == "left":
        # if one position or stategy i to take first t from the left

        t_val = utils.linspace_value(argv.tmin, argv.tmax, argv.state_size, t_pos[0])
    elif argv.strategy == "right":
        # if  stategy i to take first t from the right

        t_val = utils.linspace_value(argv.tmin, argv.tmax, argv.state_size, t_pos[-1])
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
        t_val = utils.linspace_value(argv.tmin, argv.tmax, argv.state_size,
                                     int(max_end_i-max_so_far//2))

    if t_vals is not None and argv.strategy == "middle_cones":
        # if strategy is to take the middle of the nearest higher and lower previous t

        prevt_l = t_vals[t_vals < t_val].copy()
        if prevt_l.shape[0] == 0:
            prevt_l = 0
        else:
            prevt_l = np.max(prevt_l)

        prevt_r = t_vals[t_vals > t_val].copy()
        if prevt_r.shape[0] == 0:
            prevt_r = 1
        else:
            prevt_r = np.min(prevt_r)

        t_val = (prevt_l+prevt_r)/2

    return t_val, utils.linspace_value(1, 0, argv.state_size, fm_pos)

def add_cone(state, conf_mat, t_val, argv):
    """ add CONE step on state """

    x_cone = np.linspace(argv.tmin, argv.tmax, argv.state_size)
    y_cone = np.linspace(1, 0, argv.state_size)

    curr_fm, _, slopel, sloper, offsetl, offsetr = get_slope(conf_mat, t_val,
                                                             "cone", beta=argv.beta)

    timer = time()
    grid = np.meshgrid(x_cone, y_cone)
    sup_linel = grid[1]-slopel*grid[0]-offsetl >= 0
    sup_liner = grid[1]-sloper*grid[0]-offsetr >= 0
    log.debug("\t\ttimer meshgrid: %f", time()-timer)

    timer = time()
    state[np.logical_and(sup_linel, sup_liner)] = -argv.state_size
    log.debug("\t\ttimer draw: %f", time()-timer)

    return curr_fm

def get_slope(conf_mat, t_val, algo, eps1=0, beta=1.0):
    """ return all values corresponding to a cone  (fm, phi, slopel, sloper, offsetl, offsetr) """

    curr_fm = utils.micro_fmeasure(conf_mat, beta)
    nb_pos = conf_mat[1:].sum()

    if algo.lower() == "cone":
        nb_fp = conf_mat[0, 1:].sum()
        nb_fn = conf_mat[1:].sum()-np.diagonal(conf_mat[1:, 1:]).sum()

        phi = 1/(1+beta**2)*nb_pos - nb_fn + nb_fp

        if conf_mat.shape[0] == 2:
            slopel, sloper = bin_slope(conf_mat, beta)
        else:
            slopel, sloper = mclass_slope(conf_mat, beta)
    elif algo.lower() == "parambath":
        phi = 1/nb_pos
        m_norm = math.sqrt(conf_mat[0].sum()**2+nb_pos**2)
        sloper = 2*2*phi*m_norm
        slopel = -sloper
    else:
        log.error("Unknown bound algorithm %s, only cone or parambath available.", algo)

    offsetl = curr_fm-slopel*t_val+eps1*phi
    offsetr = curr_fm-sloper*t_val+eps1*phi

    return curr_fm, phi, slopel, sloper, offsetl, offsetr

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
        lin_coeff = np.full_like(e_vect, -1)
        lin_coeff[0] = 1
    else:
        lin_coeff = np.full_like(e_vect, 1)
        lin_coeff[0] = -1

    tp_vect = conf_mat.sum(axis=1)-np.diagonal(conf_mat)

    bounds = [(-e_vect[cl_i], tp_vect[cl_i]) for cl_i in range(nb_class)]

    ineq_coeff = np.ones_like(e_vect)
    ineq_coeff[1:] = (beta**2*conf_mat[1:].sum()+e_vect[0])/(tp_vect[1:].sum()+10**(-9))

    res = linprog(lin_coeff, A_ub=[ineq_coeff], b_ub=[[0]], bounds=bounds)

    return res.x[0]-res.x[1:].sum()+e_vect[0]-e_vect[1:].sum()

def compute_weights(t_value, nb_class, beta=1.0):
    """ Compute class weights """

    weights = {class_i: (1+beta**2-t_value) for class_i in range(1, nb_class)}
    weights[0] = t_value

    return weights

def state_to_png(state, argv, cone_i=None):
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

        skimage.io.imsave("%s/state_cone%d.png"%(argv.save_states, cone),
                          skimage.transform.resize((state_mat+argv.state_size)/(argv.state_size+1),
                                                   (1000, 1000)))

def find_next_cone(saved_cones, cones_t, possible_next_t, possible_next_fm, tmin=0, tmax=1):
    """ find next t without state """

    double_fm = np.repeat(saved_cones[:-1, [0]], 2, axis=1)

    # all (t, fm) couple of all the intesections of the last cone with the other cones and y=1
    for side in [2, 3]:
        t_inter = ((saved_cones[:-1, [4, 5]]-saved_cones[-1, side+2])
                   /(saved_cones[-1, side]-saved_cones[:-1, [2, 3]]))

        # Handle specific case
        t_inter = np.nan_to_num(t_inter)
        t_inter[t_inter[:, 1] == 0][:, 1] = 1

        t_inter = np.clip(t_inter, tmin, tmax)

        fm_inter = (t_inter*saved_cones[-1, side]+saved_cones[-1, side+2])

        # keep only the couples with fm > fm of cone
        keep_inter = (fm_inter >= saved_cones[-1, 0])*(fm_inter >= double_fm)

        possible_next_t = np.concatenate([possible_next_t, t_inter[keep_inter].reshape(-1)])
        possible_next_fm = np.concatenate([possible_next_fm, fm_inter[keep_inter].reshape(-1)])

    out_cones = np.zeros_like(possible_next_t, dtype=np.bool)

    # test each couple if it is at least in one cone
    for inter_i, (t_inter, fm_inter) in enumerate(zip(possible_next_t, possible_next_fm)):
        out_cones[inter_i] = ((fm_inter-saved_cones[1:, 2]*t_inter-saved_cones[1:, 4] <= 10**(-9))
                              +(fm_inter-saved_cones[1:, 3]*t_inter-saved_cones[1:, 5] <= 10**(-9))).all()

    possible_next_t = possible_next_t[out_cones]
    possible_next_fm = possible_next_fm[out_cones]

    inter_i = np.argmax(possible_next_fm)

    prevt_l = cones_t[cones_t < possible_next_t[inter_i]].copy()
    if prevt_l.shape[0] == 0:
        prevt_l = 0
    else:
        prevt_l = np.max(prevt_l)

    prevt_r = cones_t[cones_t > possible_next_t[inter_i]].copy()
    if prevt_r.shape[0] == 0:
        prevt_r = 1
    else:
        prevt_r = np.min(prevt_r)

    return (prevt_l+prevt_r)/2, possible_next_fm[inter_i], possible_next_t, possible_next_fm

def outputs_to_json(file_path, dataset, classif, c_value=None):
    """ Create json file for visualization"""

    file_path = file_path.format(dataset, classif)

    results = np.load(file_path).item()

    if c_value is not None and not isinstance(c_value, list):
        c_value = [c_value]
    elif c_value is None:
        c_value = results.keys()


    for c_val in c_value:
        conf_mats = results[c_val]["confusions"]["train"]

        to_json = {"points":[], "P":int(conf_mats[0][1].sum()), "N":int(conf_mats[0][0].sum()),
                   "beta":1.0, "C":c_val, "dataset":dataset, "classif":classif}

        for conf_i, conf_mat in enumerate(conf_mats):
            to_json["points"].append({"t":float(results[c_val]["t_values"][conf_i]),
                                      "fp":int(conf_mat[0, 1]),
                                      "fn":int(conf_mat[1, 0])})

        utils.write_json("%s_C%s.json"%(file_path.replace(".npy", ""), str(c_val).replace(".", "")), to_json)
