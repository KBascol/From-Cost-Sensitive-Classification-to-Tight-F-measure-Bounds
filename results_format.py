"""
    Scripts managing specific plots
"""

import os
import sys
import math
import logging as log

import numpy as np
import matplotlib.pyplot as plt

import cone, utils

def draw_cones(result_file, c_val, algo, secondary_file="", ymax=1, ymin=0, xmax=1, xmin=0,
               beta=1.0, eps1=0, save=True, subset="train"):
    """ draw cones corresponding to given results and bounds """

    results = np.load(result_file).item()[c_val]
    conf_mats = results["confusions"][subset]
    t_values = results["t_values"]

    fmeas = []
    plt.figure(figsize=(7, 7))

    for x_curr, conf in zip(t_values, conf_mats):
        y_curr, phi, slopel, sloper, offsetl, offsetr = cone.get_slope(conf, x_curr, algo=algo,
                                                                       eps1=eps1, beta=beta)
        fmeas.append(y_curr)

        if slopel == 0:
            x_left = -1000
        else:
            x_left = (1-offsetl)/slopel

        if sloper == 0:
            x_right = 1000
        else:
            x_right = (1-offsetr)/sloper

        y_center = y_curr+eps1*phi

        plt.fill([x_left, x_curr, x_right], [1, y_center, 1], c="#33333380")

    plt.scatter(t_values, fmeas, zorder=50, c="#005599")

    if secondary_file:
        sec_results = np.load(secondary_file).item()[c_val]
        sec_conf_mats = sec_results["confusions"][subset]
        sec_t_values = sec_results["t_values"]

        plt.scatter(sec_t_values, [utils.micro_fmeasure(conf_mat) for conf_mat in sec_conf_mats],
                    marker="x", zorder=100, c="k")
    axes = plt.gca()

    axes.set_xticks(t_values, minor=True)
    axes.tick_params(labelsize=14)

    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.ylabel("F-Measure", fontsize=16)
    plt.xlabel("t", fontsize=16)

    if save:
        plt.savefig("%s_draw%s_c%f.pdf"%(result_file.replace(".npy", ""), algo, c_val), dpi=1000,
                    framon=False, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.show()

def get_results_fm(result_file, nb_steps, folds=None, beta=1.0, tune_thresh=False, dataset_path=""):
    """ return best svm in validation on test set for CONE method"""

    if folds is None:
        folds = range(5)

    fmeas = np.zeros(len(folds), dtype=np.float64)

    if tune_thresh:
        assert os.path.isfile(dataset_path), "Dataset required when tuning threashold"

        dataset = np.load(dataset_path)["dataset"].item()

    for fold_i, fold in enumerate(folds):
        valid_fm = np.zeros(4, dtype=np.float64)

        results = np.load(result_file%fold).item()

        if "cone" in result_file.lower():
            conf_grid = list(range(nb_steps))
        elif "parambath" in result_file.lower():
            max_index = results[list(results.keys())[0]]["t_values"].shape[0]-1

            space_grid = max_index/(nb_steps+1)
            conf_grid = [int(space_grid*i) for i in range(1, nb_steps+1)]
        else:
            conf_grid = [0]

        for c_val in results:
            c_results = results[c_val]

            if tune_thresh and "predictions" not in c_results:
                log.error("Predictions needed when tuning threshold.")
                sys.exit(0)

            for conf_i in conf_grid:
                if "predictions" not in c_results:
                    preds = None
                    labels = None
                else:
                    preds = c_results["predictions"]["valid"]
                    labels = dataset["fold%d"%fold_i]["valid"]["labels"]

                fm_tmp, thres_tmp = utils.comp_fm(c_results["confusions"]["valid"][conf_i], beta,
                                                  tune_thresh, preds, labels)

                if fm_tmp >= valid_fm[0]:
                    valid_fm = [fm_tmp, c_val, c_results["t_values"][conf_i], thres_tmp]

        c_results = results[valid_fm[1]]
        t_index = np.where(c_results["t_values"] == valid_fm[2])[0]

        if t_index.shape[0] != 0:
            t_index = t_index[0]
        else:
            t_index = 0

        if tune_thresh:
            test_conf = utils.thresh_conf(c_results["predictions"]["test"][t_index],
                                          dataset["fold%d"%fold_i]["test"]["labels"],
                                          valid_fm[3])
        else:
            test_conf = c_results["confusions"]["test"][t_index]

        fmeas[fold_i] = utils.micro_fmeasure(test_conf, beta)

    fmeas_mean = fmeas.mean()*100
    std_dev = np.sqrt(fmeas.var())*100

    log.info("AVERAGE: %f STANDARD DEV: %f", fmeas_mean, std_dev)

    return fmeas_mean, std_dev

def paper_table(base_dir, dataset_order, methods_order, beta=1.0,
                nb_steps=19, tune_thresh=False, folds=None, save_file=False):
    """ take a dictionary containning all logs paths and retrieve a latex table """

    table = np.full((len(dataset_order), len(methods_order), 2), -1, dtype=np.float32)

    hhline = "="*(len(methods_order)+1)

    for dataset_i, dataset in enumerate(dataset_order):
        for meth_i, meth in enumerate(methods_order):
            cell = table[dataset_i, meth_i]

            file_format = (base_dir, dataset, *meth)
            cell[0], cell[1] = get_results_fm("{0}/{1}/{2}_{3}_{0}_fold%d.pny".format(*file_format),
                                              nb_steps, folds, beta, tune_thresh)

    tabular = "\\begin{tabular}{%s}\n \\hline\n"
    columns = "|c||"
    header = "Datasets"
    for meth, classif in methods_order:
        columns += "c|"

        if classif.lower() == "logi_reg":
            header += " & LR"
        elif classif.lower() == "linear_svm":
            header += " & SVM"

        if meth.lower() == "ir":
            header += "$_{I.R.}$"
        elif meth.lower() == "parambath":
            header += "$_{P}$"
        elif meth.lower() == "bis":
            header += "$_{B}$"
        elif meth.lower() == "cone":
            header += "$_{C}$"

            if classif == "liblinear":
                columns += "|"

    header += ("\\\\\n \t\\hhline{|%s|} \n"%hhline)

    latex_table = tabular%columns
    latex_table += header

    for name, res_data in zip(dataset_order, table):
        row = "\t%s"%name[0]

        for meth_i, res_meth in enumerate(res_data):
            row += " & "
            if res_meth[0] < 10:
                row += "\\phantom{0}"
            row += "%.1f "%res_meth[0]

            if res_meth[1] < 10:
                row += "{\\tiny \\phantom{0}}"
            row += "\\stdev{%.1f}"%res_meth[1]
        latex_table += row + " \\\\ \n"

        if name[0] not in ["News20", "Wine"]:
            latex_table += "\t\\hline \n"
        else:
            latex_table += "\t\\hhline{|%s|} \n"%hhline

    row = "\t Average"
    for meth_i, res_meth in enumerate(table.mean(axis=0)):
        row += " & "
        if res_meth[0] < 10:
            row += "\\phantom{0}"
        row += "%.1f "%res_meth[0]

        if res_meth[1] < 10:
            row += "{\\tiny \\phantom{0}}"
        row += "\\stdev{%.1f}"%res_meth[1]
    latex_table += row + " \\\\ \n"
    latex_table += "\t\\hline \n"
    latex_table += "\\end{tabular}\n"

    if save_file:
        with open("%s/table_%d_cones.tex"%(base_dir, nb_steps), "w") as table_file:
            table_file.write(latex_table)
    else:
        print(latex_table)

def bounds_vs_eps1(result_file, c_val=1, t_min=0, t_max=1, subset="train", ymin=0, ymax=1.1,
                   tspace_size=10000, max_nb_eps=1000, beta=1.0, thresh_max=1, save_fig=True):
    """ plot cone and parambath bounds from the save set of points in function of epsilon1 """


    results = np.load(result_file).item()
    conf_mats = results[c_val]["confusions"][subset]
    t_values = results[c_val]["t_values"]

    nb_exemples = int(conf_mats[0].sum())
    nb_values = min([nb_exemples, max_nb_eps])

    eps_vals = np.linspace(0, nb_exemples, nb_values).astype(int)

    bound_p = np.ones(nb_values)
    bound_c = np.ones(nb_values)

    tspace = np.linspace(t_min, t_max, tspace_size)

    max_fm = max([utils.micro_fmeasure(conf, beta) for conf in conf_mats])

    for epsilon_i, epsilon in enumerate(eps_vals):
        log.debug("Epsilon1 %d/%d", epsilon, nb_exemples)

        tmp_bounds_p = np.zeros((tspace_size, conf_mats.shape[0]))
        tmp_bounds_c = np.zeros((tspace_size, conf_mats.shape[0]))

        for conf_i, conf in enumerate(conf_mats):
            t_val = t_values[conf_i]

            t_inf = tspace < t_val
            t_sup = tspace > t_val
            t_eq = tspace == t_val

            fmeas, phi, slopel, sloper, offl, offr = cone.get_slope(conf, t_val, "parambath",
                                                                    eps1=epsilon, beta=beta)
            tmp_bounds_p[:, conf_i] += t_inf.astype(int)*(tspace*slopel+offl)\
                                       + t_sup.astype(int)*(tspace*sloper+offr)\
                                       + t_eq.astype(int)*(fmeas+phi*epsilon)

            fmeas, phi, slopel, sloper, offl, offr = cone.get_slope(conf, t_val, "cone",
                                                                    eps1=epsilon, beta=beta)
            tmp_bounds_c[:, conf_i] += t_inf.astype(int)*(tspace*slopel+offl)\
                                       + t_sup.astype(int)*(tspace*sloper+offr)\
                                       + t_eq.astype(int)*(fmeas+phi*epsilon)

        bound_p[epsilon_i] = np.min(tmp_bounds_p, axis=1).max()
        bound_c[epsilon_i] = np.min(tmp_bounds_c, axis=1).max()

        log.debug("Bgrid: %f Bcone: %f", bound_p[epsilon_i], bound_c[epsilon_i])

    if thresh_max > 0:
        bound_p[bound_p > thresh_max] = thresh_max
        bound_c[bound_c > thresh_max] = thresh_max

    plt.figure(figsize=(7, 7))
    plt.plot(eps_vals, bound_p, linestyle='-', linewidth=3, label="Our Bound")
    plt.plot(eps_vals, bound_c, linestyle='-.', linewidth=3, label="Parambath et al. (2014) Bound")
    plt.plot(eps_vals, max_fm, linestyle=':', linewidth=3, label="Best classifier F-M")

    plt.ylim(ymin, ymax)

    if thresh_max > 0:
        # trim x axis 5 pts after the last bound reach the given maximum value
        xmax = max([eps_vals[min([(bound_c < thresh_max).sum()+5, nb_values-1])],
                    eps_vals[min([(bound_p < thresh_max).sum()+5, nb_values-1])]])
    else:
        xmax = eps_vals[-1]

    plt.xlim(0, xmax)
    plt.legend(fontsize=14)
    plt.ylabel("F-Measure", fontsize=16)
    plt.xlabel("$\\epsilon_1$", fontsize=16)
    plt.grid()

    if save_fig:
        plt.savefig("%s/bound_vs_eps1.pdf"%(os.path.basename(result_file)), dpi=1000,
                    framon=False, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.show()

def fm_vs_nbclassif(root_dir, dataset, max_step=19, classif="linear_svm", bounds=False,
                    baseline=True, baseline_ir=True, folds=None, values_subset="test", valid_subset="valid",
                    beta=1.0, eps1=0, t_min=0, t_max=1, ymin=0, ymax=1.1, save_fig=True):
    """ plot graph F-Measure in function of number of cones """

    if folds is None:
        folds = range(5)

    fmeas_p = np.zeros((len(folds), max_step))
    fmeas_c = np.zeros((len(folds), max_step))

    if bounds:
        bound_p = np.ones((len(folds), max_step))
        bound_c = np.ones((len(folds), max_step))

    if baseline:
        fmeas_base = np.zeros((len(folds)))
    if baseline_ir:
        fmeas_ir = np.zeros((len(folds)))

    for fold_i in folds:
        log.debug("fold %d", fold_i)

        best_fm_p = np.zeros(2, dtype=np.float32)
        best_fm_c = np.zeros(2, dtype=np.float32)

        file_path = "{0}/{1}/%s_{2}_{1}_fold{3}.npy".format(root_dir, dataset, classif, fold_i)

        para_res = np.load(file_path%"parambath").item()
        cone_res = np.load(file_path%"cone").item()

        assert len(cone_res) == len(para_res), "Not the same C in cone and parambath"
        assert not [c_val for c_val in cone_res if c_val not in para_res], "C in cone not in parambath"

        if baseline:
            base_res = np.load(file_path%"baseline").item()
            assert len(cone_res) == len(base_res), "Not the same C in cone and baseline"
            assert not [c_val for c_val in cone_res if c_val not in base_res], "C in cone not in baseline"
            best_fm_b = np.zeros(2, dtype=np.float32)

        if baseline_ir:
            ir_res = np.load(file_path%"ir").item()
            assert len(cone_res) == len(ir_res), "Not the same C in cone and ir"
            assert not [c_val for c_val in cone_res if c_val not in ir_res], "C in cone not in ir"
            best_fm_i = np.zeros(2, dtype=np.float32)

        # Search best C for each method
        for c_val in cone_res:
            p_conf_mats = para_res[c_val]["confusions"][valid_subset]
            c_conf_mats = cone_res[c_val]["confusions"][valid_subset]

            for t_val_i in range(max_step):
                tmp_fm = utils.micro_fmeasure(p_conf_mats[t_val_i], beta)
                if tmp_fm >= best_fm_p[0]:
                    best_fm_p = [tmp_fm, c_val]

                tmp_fm = utils.micro_fmeasure(c_conf_mats[t_val_i], beta)
                if tmp_fm >= best_fm_c[0]:
                    best_fm_c = [tmp_fm, c_val]

            if baseline:
                tmp_fm = utils.micro_fmeasure(base_res[c_val]["confusions"][valid_subset][0], beta)
                if tmp_fm >= best_fm_b[0]:
                    best_fm_b = [tmp_fm, c_val]
            if baseline_ir:
                tmp_fm = utils.micro_fmeasure(ir_res[c_val]["confusions"][valid_subset][0], beta)
                if tmp_fm >= best_fm_i[0]:
                    best_fm_i = [tmp_fm, c_val]

        if baseline:
            fmeas_base[fold_i] = utils.micro_fmeasure(base_res[best_fm_b[1]]["confusions"][values_subset], beta)
        if baseline_ir:
            fmeas_ir[fold_i] = utils.micro_fmeasure(ir_res[best_fm_i[1]]["confusions"][values_subset], beta)

        t_vals_p = para_res[best_fm_p[1]]["t_values"]
        conf_para = para_res[best_fm_p[1]]["confusions"][values_subset]

        t_vals_c = cone_res[best_fm_c[1]]["t_values"]
        conf_cone = cone_res[best_fm_c[1]]["confusions"][values_subset]

        # Compute bounds for cone and parambath methods
        for nb_cones in range(1, max_step+1):
            log.debug("\tnb_cone: %d", nb_cones)

            space_grid = 19/(nb_cones+1)
            subgrid = [int(space_grid*i) for i in range(1, nb_cones+1)]
            t_vals_p_subgrid = t_vals_p[subgrid]
            conf_para_subgrid = conf_para[subgrid]

            if bounds:
                tspace = np.linspace(t_min, t_max, 10000)

                tmp_bounds_c = np.zeros((10000, nb_cones))
                tmp_bounds_p = np.zeros((10000, nb_cones))

            tmp_fmeas_p = np.zeros(nb_cones)
            tmp_fmeas_c = np.zeros(nb_cones)

            for conf_i, (conf_p, conf_c) in enumerate(zip(conf_para_subgrid,
                                                          conf_cone[range(0, nb_cones)])):
                t_val = t_vals_p_subgrid[conf_i]
                fmeas, phi, slopel, sloper, offl, offr = cone.get_slope(conf_p, t_val, "parambath",
                                                                        eps1=eps1, beta=beta)
                tmp_fmeas_p[conf_i] = fmeas

                if bounds:
                    t_inf = tspace < t_val
                    t_sup = tspace > t_val
                    t_eq = tspace == t_val
                    tmp_bounds_p[:, conf_i] += t_inf.astype(int)*(tspace*slopel+offl)\
                                               + t_sup.astype(int)*(tspace*sloper+offr)\
                                               + t_eq.astype(int)*(fmeas+phi*eps1)

                if conf_c.sum() == 0:
                    # happend when cone reached the stopping criterium
                    tmp_fmeas_c[conf_i] = tmp_fmeas_c[conf_i-1]

                    if bounds:
                        tmp_bounds_c[:, conf_i] = tmp_bounds_c[:, conf_i-1]
                else:
                    t_val = t_vals_c[conf_i]

                    fmeas, phi, slopel, sloper, offl, offr = cone.get_slope(conf_c, t_val, "cone",
                                                                            eps1=eps1, beta=beta)
                    tmp_fmeas_c[conf_i] = fmeas

                    if bounds:
                        t_inf = tspace < t_val
                        t_sup = tspace > t_val
                        t_eq = tspace == t_val
                        tmp_bounds_c[:, conf_i] += t_inf.astype(int)*(tspace*slopel+offl)\
                                                   + t_sup.astype(int)*(tspace*sloper+offr)\
                                                   + t_eq.astype(int)*(fmeas+phi*eps1)

            fmeas_p[fold_i, nb_cones-1] = tmp_fmeas_p.max()
            fmeas_c[fold_i, nb_cones-1] = tmp_fmeas_c.max()

            if bounds:
                bound_p[fold_i, nb_cones-1] = np.min(tmp_bounds_p, axis=1).max()
                bound_c[fold_i, nb_cones-1] = np.min(tmp_bounds_c, axis=1).max()

        if bounds:
            log.info("\t Bpara: %f Bcone: %f", bound_p[fold_i, -1], bound_c[fold_i, -1])


    plt.figure(figsize=(7, 7))

    labels = ["Parambath et al. (2014) F-M", "Our F-M"]

    if bounds:
        labels.insert(1, "Parambath et al. (2014) Bound")
        labels.append("Our Bound")

    if baseline_ir:
        labels.insert(0, "Baseline I.R. F-M")
    if baseline:
        labels.insert(0, "Baseline F-M")

    x_space = np.linspace(1, max_step, max_step)

    if baseline:
        plt.plot(x_space, np.repeat(fmeas_base.mean(), max_step), c="#000000", linewidth=3)
    if baseline_ir:
        plt.plot(x_space, np.repeat(fmeas_ir.mean(), max_step), c="#888888", linewidth=3)

    plt.plot(x_space, fmeas_p.mean(axis=0), c="#FF0000", linestyle='-.', linewidth=3)
    if bounds:
        plt.plot(x_space, np.minimum(bound_p.mean(axis=0), 1), c="#AA5500", linestyle='-',
                 marker="+", linewidth=3, mew=3)

    plt.plot(x_space, fmeas_c.mean(axis=0), c="#0000FF", linestyle=':', linewidth=3)
    if bounds:
        plt.plot(x_space, np.minimum(bound_c.mean(axis=0), 1), c="#008899", linestyle='--',
                 linewidth=3)

    plt.legend(labels, fontsize=14)
    # if legend is over the curves :
    # plt.legend(labels, bbox_to_anchor=(0., 1.02, 1., .102),
    #            frameon=False, loc=3, ncol=3, fontsize=14)

    axes = plt.gca()
    axes.tick_params(labelsize=14)
    plt.ylim(ymin, ymax)
    plt.xticks(list(range(1, max_step, 3))+[max_step])
    plt.ylabel("F-Measure", fontsize=16)
    plt.xlabel("Number of steps / Grid size", fontsize=16)

    file_name = "%s/%s/fm_"%(root_dir, dataset)
    if bounds:
        file_name += "and_bounds_"
    file_name += "vs_nbclassif"

    if baseline:
        file_name += "_wth_base"
    if baseline:
        file_name += "_wth_ir"

    if save_fig:
        plt.savefig(file_name+".pdf", dpi=1000, framon=False, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.show()
