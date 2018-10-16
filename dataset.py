"""
    Create dataset file from different sources
"""

import os

import numpy as np

np.random.seed(42000)

def letter(filename):
    """
        Create npz file containing Letter dataset
        source: https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
    """

    labels = set()
    nb_class = 26
    data = []

    nb_value = 0
    with open(filename, "r") as file:
        for line in file.readlines():
            split_line = line.split(",")

            label = ord(split_line[0].lower())-97
            labels.add(label)

            data.append([label]+list(map(int, split_line[1:])))

            nb_value += 1

    data_to_npz(np.array(data), nb_class, os.path.basename(filename), "Letter")

def get_adult_features(filename, feat_count):
    """ Change values from Adult dataset to numerical vales """
    data = []

    with open(filename, "r") as file:
        for line in file.readlines():
            split_line = line.split(", ")

            if len(split_line) != 15:
                break

            data_tmp = []

            data_tmp.append(int(">" in split_line[14]))
            data_tmp.append(int(split_line[0]))
            data_tmp.append(get_feat(feat_count, "wclass", split_line[1]))
            data_tmp.append(int(split_line[2]))
            data_tmp.append(get_feat(feat_count, "edu", split_line[3]))
            data_tmp.append(int(split_line[4]))
            data_tmp.append(get_feat(feat_count, "marit", split_line[5]))
            data_tmp.append(get_feat(feat_count, "occ", split_line[6]))
            data_tmp.append(get_feat(feat_count, "relat", split_line[7]))
            data_tmp.append(get_feat(feat_count, "race", split_line[8]))
            data_tmp.append(get_feat(feat_count, "sex", split_line[13]))
            data_tmp.append(int(split_line[10]))
            data_tmp.append(int(split_line[11]))
            data_tmp.append(int(split_line[12]))
            data_tmp.append(get_feat(feat_count, "count", split_line[9]))

            data.append(data_tmp)

    return adult_to_hotvect(data)

def get_feat(feat_count, feat, value):
    """ Change string value to numerical value """

    if value not in feat_count[feat][1]:
        feat_count[feat][0] += 1
        feat_count[feat][1][value] = feat_count[feat][0]

    return feat_count[feat][1][value]

def adult_to_hotvect(data):
    """ 14 feature Adult to one hot vectors concatenation """

    vector = np.zeros((data.shape[0], 124))
    vector[:, 0] = data[:, 0]

    for data_i, vect_i in [(1, 1), (3, 13), (5, 33), (11, 71), (12, 75), (13, 79)]:
        tmp_data = data[:, data_i]
        quantiles = np.percentile(tmp_data[tmp_data != -1], [0, 25, 50, 75])

        for ex_i in range(data.shape[0]):
            quant = np.where(data[ex_i, data_i] >= quantiles)[0].max()
            vector[ex_i, vect_i+quant] = 1

    for ex_i in range(data.shape[0]):
        for data_i, vect_i in [(2, 5), (4, 17), (6, 37), (7, 44), (8, 58),
                               (9, 64), (10, 69), (14, 83)]:
            if data[ex_i, data_i] != -1:
                vector[ex_i, vect_i+data[ex_i, data_i]] = 1

    return vector

def adult(filename):
    """
        Create npz file containing Adult dataset
        source: https://archive.ics.uci.edu/ml/datasets/Adult
    """

    feat_count = {"wclass":[-1, {"?":-1}], "edu":[-1, {"?":-1}], "marit":[-1, {"?":-1}],
                  "occ":[-1, {"?":-1}], "relat":[-1, {"?":-1}], "race":[-1, {"?":-1}],
                  "count":[-1, {"?":-1}], "sex":[1, {"?":-1, "Female":0, "Male":1}]}

    nb_class = 2

    data = adult_to_hotvect(np.array(get_adult_features(filename+".data", feat_count)))

    test = adult_to_hotvect(np.array(get_adult_features(filename+".test", feat_count)))

    data_to_npz(data, nb_class, os.path.basename(filename), "Adult", test=test)

def news20(filename):
    """
        Create npz file containing News20 dataset
        source: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#news20 (scaled)
    """

    nb_class = 20

    data = np.zeros((15935, 62062), dtype=np.float32)
    with open(filename+".train", "r") as file:
        for line_i, line in enumerate(file.readlines()):
            split_line = line.split(" ")

            data[line_i, 0] = float(split_line[0])-1

            for value in split_line[1:]:
                if len(value) > 1:
                    split_val = value.split(":")
                    index = int(split_val[0])
                    data[line_i, index] = float(split_val[1])

    test = np.zeros((3993, 62062), dtype=np.float32)
    with open(filename+".test", "r") as file:
        for line_i, line in enumerate(file.readlines()):
            split_line = line.split(" ")

            test[line_i, 0] = float(split_line[0])-1

            for value in split_line[1:]:
                if len(value) > 1:
                    split_val = value.split(":")
                    index = int(split_val[0])
                    test[line_i, index] = float(split_val[1])

    data_to_npz(data, nb_class, os.path.basename(filename), "News20", test=test)

def ijcnn(filename):
    """
       Create npz file containing IJCNN dataset
       source: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1
    """

    nb_class = 2

    nb_value = 49990
    data = np.zeros((nb_value, 23), dtype=np.float32)
    with open(filename, "r") as file:
        for line_i, line in enumerate(file.readlines()):
            split_line = line.split(" ")

            data[line_i, 0] = int(split_line[0] != "-1")

            for value in split_line[1:]:
                if len(value) > 1:
                    val_split = value.split(":")
                    index = int(val_split[0])
                    data[line_i, index] = float(val_split[1])


    nb_value = 91701
    test = np.zeros((nb_value, 23), dtype=np.float32)
    with open(filename+".t", "r") as file:
        for line_i, line in enumerate(file.readlines()):
            split_line = line.split(" ")

            test[line_i, 0] = int(split_line[0] != "-1")

            for value in split_line[1:]:
                if len(value) > 1:
                    val_split = value.split(":")
                    index = int(val_split[0])
                    test[line_i, index] = float(val_split[1])

    data_to_npz(data, nb_class, os.path.basename(filename), "IJCNN", test=test)

def wine(filename):
    """
        Create npz file containing Wine dataset
        source: https://archive.ics.uci.edu/ml/datasets/Wine
    """

    nb_class = 2

    nb_data = 1599
    data = np.zeros((nb_data, 12), dtype=np.float32)
    with open(filename, "r") as file:
        ignored = 0
        for line_i, line in enumerate(file.readlines()):
            if line and line[0] != "@":
                line_i -= ignored
                split_line = line.split(",")

                data[line_i, 0] = int("positive" in split_line[-1])

                for index, value in enumerate(split_line[:-1]):
                    data[line_i, index+1] = float(value)
            else:
                ignored += 1

    data_to_npz(data, nb_class, os.path.basename(filename), "Wine")

def yeast(filename):
    """
        Create npz file containing Yeast dataset
        source: https://archive.ics.uci.edu/ml/datasets/Yeast
    """

    nb_class = 2

    nb_data = 1484
    data = np.zeros((nb_data, 9), dtype=np.float32)
    with open(filename, "r") as file:
        ignored = 0
        for line_i, line in enumerate(file.readlines()):
            if line and line[0] != "@":
                line_i -= ignored
                split_line = line.split(", ")

                data[line_i, 0] = int("positive" in split_line[-1])

                for index, value in enumerate(split_line[:-1]):
                    data[line_i, index+1] = float(value)
            else:
                ignored += 1

    data_to_npz(data, nb_class, os.path.basename(filename), "Yeast")

def abalone(filename, pos_cl=10):
    """
        Create npz file containing Abalone dataset in binary setting
        source: https://archive.ics.uci.edu/ml/datasets/Abalone
    """

    nb_class = 2

    nb_data = 4174
    data = np.zeros((nb_data, 11), dtype=np.float32)
    with open(filename, "r") as file:
        ignored = 0
        for line_i, line in enumerate(file.readlines()):
            if line and line[0] != "@":
                line_i -= ignored
                split_line = line.split(", ")

                data[line_i, 0] = int(int(split_line[-1]) == pos_cl)

                if split_line[0] == "F":
                    data[line_i, 1] = 1
                elif split_line[0] == "M":
                    data[line_i, 2] = 1
                elif split_line[0] == "I":
                    data[line_i, 3] = 1

                for index, value in enumerate(split_line[1:-1]):
                    data[line_i, index+4] = float(value)
            else:
                ignored += 1

    data_to_npz(data, nb_class, os.path.basename(filename), "Abalone%d"%pos_cl)

def pageblocks(filename, pos_classes):
    """
        Create npz file containing Pageblocks dataset in binary setting
        source: https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification
    """

    nb_class = 2

    nb_data = 5473
    data = np.zeros((nb_data, 11), dtype=np.float32)
    with open(filename, "r") as file:
        ignored = 0
        for line_i, line in enumerate(file.readlines()):
            if line and line[0] != "@":
                line_i -= ignored
                split_line = line.split(",")

                data[line_i, 0] = int(split_line[-1]) in pos_classes

                for index, value in enumerate(split_line[:-1]):
                    data[line_i, index+1] = float(value)
            else:
                ignored += 1

    data_to_npz(data, nb_class, os.path.basename(filename), "Pageblocks")

def satimage(filename, pos_classes):
    """
        Create npz file containing Satimage dataset in binary setting
        source: https://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29
    """

    nb_class = 2

    nb_data = 6435
    data = np.zeros((nb_data, 37), dtype=np.float32)
    with open(filename, "r") as file:
        ignored = 0
        for line_i, line in enumerate(file.readlines()):
            if line and line[0] != "@":
                line_i -= ignored
                split_line = line.split(",")

                data[line_i, 0] = int(split_line[-1]) in pos_classes

                for index, value in enumerate(split_line[:-1]):
                    data[line_i, index+1] = float(value)
            else:
                ignored += 1

    data_to_npz(data, nb_class, os.path.basename(filename), "Satimage")

def data_to_npz(data, nb_class, root_dir, dataset_name, nb_folds=5, test=None):
    """ Do nb_folds random split with 1/4 all data in test (if not provided), 1/3 remaining in valid of data """

    dataset = {"nb_class": nb_class}

    for feat_i in range(data.shape[1]-1):
        if data[:, feat_i].max() != 1 and data[:, feat_i].min() != 0:
            data[:, feat_i] = ((data[:, feat_i] - data[:, feat_i].min())
                               (data[:, feat_i].max() - data[:, feat_i].min()))

    for fold_i in range(nb_folds):
        fold = {"train": {}, "valid": {}, "test": {}}

        if test is None:
            test, data = np.vsplit(data[np.random.permutation(data.shape[0])],
                                   (data.shape[0]//4))

        fold["test"]["exemples"] = test[:, 1:]
        fold["test"]["labels"] = test[:, 0].astype(int)

        valid, train = np.vsplit(data[np.random.permutation(data.shape[0])],
                                 (data.shape[0]//3))

        fold["train"]["exemples"] = train[:, 1:]
        fold["train"]["labels"] = train[:, 0].astype(int)

        fold["valid"]["exemples"] = valid[:, 1:]
        fold["valid"]["labels"] = valid[:, 0].astype(int)

        dataset["fold%d"%fold_i].append(fold)

    np.savez("%s/%s.npz"%(root_dir, dataset_name), dataset=dataset)
