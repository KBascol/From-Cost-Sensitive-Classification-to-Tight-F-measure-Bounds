# TODO save all np.savez(os.path.dirname(filename)+"/dataset.npz", dataset=dataset)
# TODO format dataset[subset]["exemples"] dataset[subset]["labels"] dataset["nb_class"]


import numpy as np


def load_dataset(dataset, fold):
    """ load dataset as folds """

    if isinstance(dataset, str):
        dataset = np.load(dataset)

        if isinstance(dataset, np.ndarray):
            dataset = dataset.item()
        else:
            dataset = dataset['dataset'].item()

    data_fold = dataset["folds"][fold]

    nb_class = max(2, dataset["nb_classes"])

    return (data_fold["train"][0],
            data_fold["train"][1].astype(int),
            data_fold["valid"][0],
            data_fold["valid"][1].astype(int),
            data_fold["test"][0],
            data_fold["test"][1].astype(int),
            nb_class)


def letter_npy(filename, nb_class=26, nb_folds=5):
    """ Create npy file containing Letter dataset with numerical values """

    labels = set()
    dataset = {"nb_classes":nb_class, "folds":[]}

    for _ in range(nb_folds):
        fold = {}
        data = []

        nb_value = 0
        with open(filename, "r") as file:
            for line in file.readlines():
                split_line = line.split(",")

                label = ord(split_line[0].lower())-97
                labels.add(label)

                data.append([label]+list(map(int, split_line[1:])))

                nb_value += 1

        nb_test = nb_value//4
        nb_valid = (nb_value-nb_test)//3

        data_tmp = []
        for _ in range(nb_test):
            data_tmp.append(data.pop(ndom.randint(0, nb_value)))
            nb_value -= 1

        data_tmp = np.array(data_tmp)
        fold["test"] = (data_tmp[:, 1:], data_tmp[:, 0])

        data_tmp = []
        for _ in range(nb_valid):
            data_tmp.append(data.pop(ndom.randint(0, nb_value)))
            nb_value -= 1

        data_tmp = np.array(data_tmp)
        fold["valid"] = (data_tmp[:, 1:], data_tmp[:, 0])

        data = np.array(data)
        fold["train"] = (data[:, 1:], data[:, 0])

        dataset["folds"].append(fold)

    np.save(os.path.dirname(filename)+"/dataset.npy", dataset)

    return dataset

def num_adult(filename, feat_count):
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

    return data

def get_feat(feat_count, feat, value):
    """ Change string value to numerical value """

    if value not in feat_count[feat][1]:
        feat_count[feat][0] += 1
        feat_count[feat][1][value] = feat_count[feat][0]

    return feat_count[feat][1][value]

def adult_npy(filename, hotvect=True):
    """ Create npy file containing Adult dataset with numerical values """

    dataset = {}
    feat_count = {"wclass":[-1, {"?":-1}], "edu":[-1, {"?":-1}], "marit":[-1, {"?":-1}],
                  "occ":[-1, {"?":-1}], "relat":[-1, {"?":-1}], "race":[-1, {"?":-1}],
                  "count":[-1, {"?":-1}], "sex":[1, {"?":-1, "Female":0, "Male":1}]}

    dataset["nb_classes"] = 1
    dataset["folds"] = []

    for _ in range(5):
        fold = {}

        data_train = np.array(num_adult(filename, feat_count))

        if hotvect:
            data_train = adult_to_hotvect(data_train)
        else:
            for col_i in range(1, 15):
                missing = data_train[:, col_i] == -1

                mean_val = data_train[np.logical_not(missing), col_i].mean()
                data_train[missing, col_i] = int(mean_val)

        nb_value = len(data_train)
        nb_valid = (nb_value)//3

        if hotvect:
            data_valid = np.zeros((nb_valid, 124))
        else:
            data_valid = np.zeros((nb_valid, 16))

        for valid_i in range(nb_valid):
            val_i = ndom.randint(0, nb_value)

            data_valid[valid_i] = data_train[val_i]
            data_train = np.delete(data_train, val_i, 0)

            nb_value -= 1

        fold["valid"] = (data_valid[:, 1:], data_valid[:, 0])

        fold["train"] = (data_train[:, 1:], data_train[:, 0])

        data_test = np.array(num_adult(filename.replace(".data", ".test"), feat_count))

        if hotvect:
            data_test = adult_to_hotvect(data_test)
        else:
            for col_i in range(1, 15):
                missing = data_test[:, col_i] == -1

                mean_val = data_test[np.logical_not(missing), col_i].mean()
                data_test[missing, col_i] = int(mean_val)

        fold["test"] = (data_test[:, 1:], data_test[:, 0])

        dataset["folds"].append(fold)

    np.save(os.path.dirname(filename)+"/dataset.npy", dataset)

    return dataset

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

def a9a_npy(filename):
    """ Create npy file containing Adult dataset with numerical values """

    data = []
    dataset = {"folds":[]}
    dataset["nb_classes"] = 1

    for _ in range(5):
        fold = {}
        nb_value = 0
        data = []
        with open(filename, "r") as file:
            for line in file.readlines():
                data_line = np.zeros(124)
                split_line = line.split(" ")

                data_line[0] = int(split_line[0] == "+1")

                for value in split_line[1:]:
                    if len(value) > 1:
                        index = int(value.split(":")[0])
                        data_line[index] = 1

                data.append(data_line)

                nb_value += 1

        nb_valid = nb_value//3

        data_tmp = []
        for _ in range(nb_valid):
            data_tmp.append(data.pop(ndom.randint(0, nb_value)))
            nb_value -= 1

        data_tmp = np.array(data_tmp, dtype=int)
        fold["valid"] = (data_tmp[:, 1:], data_tmp[:, 0])

        data = np.array(data, dtype=int)
        fold["train"] = (data[:, 1:], data[:, 0])

        data = []
        with open(filename+".t", "r") as file:
            for line in file.readlines():
                data_line = np.zeros(124)
                split_line = line.split(" ")

                data_line[0] = int(split_line[0] == "+1")

                for value in split_line[1:]:
                    if len(value) > 1:
                        index = int(value.split(":")[0])
                        data_line[index] = 1

                data.append(data_line)

        data = np.array(data, dtype=int)
        fold["test"] = (data[:, 1:], data[:, 0])

        dataset["folds"].append(fold)

    np.save(os.path.dirname(filename)+"/dataset_a9a.npy", dataset)

    return dataset

def news20_npy(filename, scaled=True):
    """ Create npy file containing Adult dataset with numerical values """

    data = []
    dataset = {"folds":[], "nb_classes":20}

    for _ in range(5):
        fold = {}
        data = None
        gc.collect()

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

        nb_valid = 15935//3

        valid, train = np.vsplit(data[np.random.permutation(data.shape[0])], (nb_valid, ))

        fold["valid"] = (sparse.csr_matrix(valid[:, 1:], dtype=np.float32), valid[:, 0].astype(int))

        fold["train"] = (sparse.csr_matrix(train[:, 1:], dtype=np.float32), train[:, 0].astype(int))

        data = np.zeros((3993, 62062), dtype=np.float32)
        with open(filename+".test", "r") as file:
            for line_i, line in enumerate(file.readlines()):
                split_line = line.split(" ")

                data[line_i, 0] = float(split_line[0])-1

                for value in split_line[1:]:
                    if len(value) > 1:
                        split_val = value.split(":")
                        index = int(split_val[0])
                        data[line_i, index] = float(split_val[1])

        fold["test"] = (sparse.csr_matrix(data[:, 1:], dtype=np.float32), data[:, 0].astype(int))

        dataset["folds"].append(fold)

    if scaled:
        np.savez(os.path.dirname(filename)+"/dataset.npz", dataset=dataset)
    else:
        np.savez(os.path.dirname(filename)+"/dataset.npz", dataset=dataset)

    return dataset


def synthetic_2D(path, nb_data=6000, noise=True):

    dataset = {"nb_classes":2, "folds":[]}

    for _ in range(5):
        fold = {}
        data = np.zeros((nb_data, 3))

        data[:, 1:] = np.random.uniform(0.0, 1.0, (nb_data, 2))
        w0_opti = np.random.uniform(0.1, 0.3)
        w1_opti = np.random.uniform(-3, -2)
        w2_opti = np.random.uniform(0.1, 1)

        data[:, 0] = (data[:, 1]*w1_opti+data[:, 2]*w2_opti+w0_opti > 0).astype(int)

        if noise:
            noise = np.random.uniform(4, 6)
            nb_pos = (data[:, 0] == 1).sum()
            to_change = np.random.uniform(0, 100, (nb_pos)) < noise
            changing = np.ones(data[data[:, 0] == 1].shape[0])
            changing[to_change] = 0
            data[data[:, 0] == 1, 0] = changing

            noise = np.random.uniform(4, 6)
            nb_neg = (data[:, 0] == 0).sum()
            to_change = np.random.uniform(0, 100, (nb_neg)) < noise
            changing = np.zeros(data[data[:, 0] == 0].shape[0])
            changing[to_change] = 1
            data[data[:, 0] == 0, 0] = changing

        q_data = nb_data//4

        test, valid, train = np.vsplit(data[np.random.permutation(data.shape[0])],
                                       (q_data, q_data+(nb_data-q_data)//3))

        fold["opti"] = (w0_opti, w1_opti, w2_opti)

        fold["test"] = (test[:, 1:], test[:, 0].astype(int))

        fold["valid"] = (valid[:, 1:], valid[:, 0].astype(int))

        fold["train"] = (train[:, 1:], train[:, 0].astype(int))

        dataset["folds"].append(fold)

    np.save(path+"/dataset_opti.npy", dataset)

def cifar_dataset(dirname):
    """ Cifar basic dataset to used format """

    dataset = {"nbClasses":10, "format":"cifar", "nbChannels":3,
               'imgDims':32, "folds":[], "nbFolds":5, "domains":None}

    for fold_path in [dirname+"/data_batch_%d"%fold_i for fold_i in range(1, 6)]:
        dataset["folds"].append(cifar_fold(fold_path))

    dataset["test"] = cifar_fold(dirname+"/test_batch")

    np.save(dirname+"/dataset.npy", dataset)

def cifar_fold(batch_path):
    """ load cifar batch file to used format """

    with open(batch_path, 'rb') as batch_file:
        fold_file = pickle.load(batch_file, encoding='bytes')

    fold = []
    for example in zip(fold_file[b'data'], fold_file[b'labels'], fold_file[b'filenames']):
        fold.append((deflatten_img(example[0])/255, example[1], str(example[2])))

    return fold

def deflatten_img(img):
    """ get image from flatten data """

    channel_size = img.shape[0]//3
    img_size = int(math.sqrt(channel_size))

    return np.dstack((img[:channel_size].reshape((img_size, img_size)),
                      img[channel_size:channel_size*2].reshape((img_size, img_size)),
                      img[channel_size*2:].reshape((img_size, img_size))))

def webspam_npy(filename):
    """ Create npy file containing Adult dataset with numerical values """

    dataset = {"folds":[]}
    dataset["nb_classes"] = 2

    nb_value = 350000
    data = np.zeros((nb_value, 255), dtype=np.float32)
    with open(filename, "r") as file:
        for line_i, line in enumerate(file.readlines()):
            split_line = line.split(" ")

            data[line_i, 0] = int(split_line[0] == "+1")

            for value in split_line[1:]:
                if len(value) > 1:
                    index = int(value.split(":")[0])
                    data[line_i, index] = 1

    for _ in range(5):
        fold = {}

        q_data = nb_value//4

        test, valid, train = np.vsplit(data[np.random.permutation(data.shape[0])],
                                       (q_data, q_data+(nb_value-q_data)//3))

        fold["test"] = (sparse.csr_matrix(test[:, 1:]), test[:, 0].astype(int))

        fold["valid"] = (sparse.csr_matrix(valid[:, 1:]), valid[:, 0].astype(int))

        fold["train"] = (sparse.csr_matrix(train[:, 1:]), train[:, 0].astype(int))

        dataset["folds"].append(fold)

    np.savez(os.path.dirname(filename)+"/dataset", dataset=dataset)

def ijcnn_npy(filename):
    """ Create npy file containing Adult dataset with numerical values """

    dataset = {"folds":[]}
    dataset["nb_classes"] = 2

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

    for _ in range(5):
        fold = {}

        valid, train = np.vsplit(data[np.random.permutation(data.shape[0])],
                                 (nb_value//3, ))

        fold["valid"] = (valid[:, 1:], valid[:, 0].astype(int))

        fold["train"] = (train[:, 1:], train[:, 0].astype(int))

        dataset["folds"].append(fold)

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

    for fold in dataset["folds"]:
        fold["test"] = (test[:, 1:], test[:, 0].astype(int))

    np.save(os.path.dirname(filename)+"/dataset.npy", dataset)

def wine_npy(filename, normalize=False):
    """ Create npy file containing Adult dataset with numerical values """

    dataset = {"folds":[]}
    dataset["nb_classes"] = 2

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

    for _ in range(5):
        fold = {}

        q_data = nb_data//4

        if normalize:
            data[:, 1:] = ((data[:, 1:]-data[:, 1:].min(axis=0, keepdims=True))
                           /(data[:, 1:].max(axis=0, keepdims=True)
                             -data[:, 1:].min(axis=0, keepdims=True)))

        test, valid, train = np.vsplit(data[np.random.permutation(data.shape[0])],
                                       (q_data, q_data+(nb_data-q_data)//3))

        fold["valid"] = (valid[:, 1:], valid[:, 0].astype(int))

        fold["train"] = (train[:, 1:], train[:, 0].astype(int))

        fold["test"] = (test[:, 1:], test[:, 0].astype(int))

        dataset["folds"].append(fold)

    path = os.path.dirname(filename)+"/dataset"

    if normalize:
        path += "_norm"

    np.save(path+"2.npy", dataset)

def yeast_npy(filename):
    """ Create npy file containing Adult dataset with numerical values """

    dataset = {"folds":[]}
    dataset["nb_classes"] = 2

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

    for _ in range(5):
        fold = {}

        q_data = nb_data//4

        test, valid, train = np.vsplit(data[np.random.permutation(data.shape[0])],
                                       (q_data, q_data+(nb_data-q_data)//3))

        fold["valid"] = (valid[:, 1:], valid[:, 0].astype(int))

        fold["train"] = (train[:, 1:], train[:, 0].astype(int))

        fold["test"] = (test[:, 1:], test[:, 0].astype(int))

        dataset["folds"].append(fold)

    np.save(os.path.dirname(filename)+"/dataset2.npy", dataset)

def abalone_npy(filename, pos_cl=16):
    """ Create npy file containing Adult dataset with numerical values """

    dataset = {"folds":[]}
    dataset["nb_classes"] = 2

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

    for _ in range(5):
        fold = {}

        q_data = nb_data//4

        test, valid, train = np.vsplit(data[np.random.permutation(data.shape[0])],
                                       (q_data, q_data+(nb_data-q_data)//3))

        fold["valid"] = (valid[:, 1:], valid[:, 0].astype(int))

        fold["train"] = (train[:, 1:], train[:, 0].astype(int))

        fold["test"] = (test[:, 1:], test[:, 0].astype(int))

        dataset["folds"].append(fold)

    np.save(os.path.dirname(filename)+"/dataset"+str(pos_cl)+".npy", dataset)

def pageblocks_npy(filename, pos_classes, normalize=False):
    """ Create npy file containing Adult dataset with numerical values """

    dataset = {"folds":[]}
    dataset["nb_classes"] = 2

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

    for _ in range(5):
        fold = {}

        q_data = nb_data//4

        if normalize:
            data[:, 1:] = ((data[:, 1:]-data[:, 1:].min(axis=0, keepdims=True))
                           /(data[:, 1:].max(axis=0, keepdims=True)
                             -data[:, 1:].min(axis=0, keepdims=True)))

        test, valid, train = np.vsplit(data[np.random.permutation(data.shape[0])],
                                       (q_data, q_data+(nb_data-q_data)//3))

        fold["valid"] = (valid[:, 1:], valid[:, 0].astype(int))

        fold["train"] = (train[:, 1:], train[:, 0].astype(int))

        fold["test"] = (test[:, 1:], test[:, 0].astype(int))

        dataset["folds"].append(fold)

    path = os.path.dirname(filename)+"/dataset"

    if normalize:
        path += "_norm"

    np.save(path+".npy", dataset)

def satimage_npy(filename, pos_classes, normalize=False):
    """ Create npy file containing Adult dataset with numerical values """

    dataset = {"folds":[]}
    dataset["nb_classes"] = 2

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

    for _ in range(5):
        fold = {}

        q_data = nb_data//4

        if normalize:
            data[:, 1:] /= 255

        test, valid, train = np.vsplit(data[np.random.permutation(data.shape[0])],
                                       (q_data, q_data+(nb_data-q_data)//3))

        fold["valid"] = (valid[:, 1:], valid[:, 0].astype(int))

        fold["train"] = (train[:, 1:], train[:, 0].astype(int))

        fold["test"] = (test[:, 1:], test[:, 0].astype(int))

        dataset["folds"].append(fold)

    path = os.path.dirname(filename)+"/dataset"

    if normalize:
        path += "_norm"

    np.save(path+".npy", dataset)

def add_const_col(dataset_path, const):
    dataset = np.load(dataset_path)
    npy = False

    if isinstance(dataset, np.ndarray):
        dataset = dataset.item()
        npy = True
    else:
        dataset = dataset['dataset'].item()

    for fold in dataset["folds"]:
        fold["train"] = (np.hstack([fold["train"][0],
                                    np.full((fold["train"][0].shape[0], 1), const)]),
                         fold["train"][1])
        fold["valid"] = (np.hstack([fold["valid"][0],
                                    np.full((fold["valid"][0].shape[0], 1), const)]),
                         fold["valid"][1])
        fold["test"] = (np.hstack([fold["test"][0],
                                   np.full((fold["test"][0].shape[0], 1), const)]),
                        fold["test"][1])


    new_dataset_path = dataset_path.replace(".np", "_col%0.f.np"%const)
    if npy:
        np.save(new_dataset_path, dataset)
    else:
        np.savez(new_dataset_path[:-4], dataset=dataset)

def pydataset_to_liblinear(dataset_path, integers):
    dataset = np.load(dataset_path)

    if isinstance(dataset, np.ndarray):
        dataset = dataset.item()
        libdata_root = dataset_path.replace(".npy", "")
    else:
        dataset = dataset['dataset'].item()
        libdata_root = dataset_path.replace(".npz", "")

    for fold_i, fold in enumerate(dataset["folds"]):
        libdata_path = libdata_root + "_liblin_f%d"%fold_i
        with open(libdata_path+".train", "w") as file:
            file.write(pydata_to_liblinear(fold["train"], integers))
        with open(libdata_path+".valid", "w") as file:
            file.write(pydata_to_liblinear(fold["valid"], integers))
        with open(libdata_path+".test", "w") as file:
            file.write(pydata_to_liblinear(fold["test"], integers))

def pydata_to_liblinear(data, integers=False):
    libdata = ""

    for example, label in zip(data[0], data[1]):
        libdata += str(int(label))

        for value_i, value in enumerate(example):
            if value != 0:
                if integers:
                    libdata += " %d:%d"%(value_i+1, value)
                else:
                    libdata += " %d:%f"%(value_i+1, value)
        libdata += "\n"

    return libdata
