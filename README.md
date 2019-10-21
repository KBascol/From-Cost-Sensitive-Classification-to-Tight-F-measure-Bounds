# From Cost-Sensitive Classification to Tight F-measure Bounds

### BibTeX
```
@inproceedings{bascol2019fmeasure,
  TITLE = {{From Cost-Sensitive Classification to Tight F-measure Bounds}},
  AUTHOR = {Bascol, Kevin and Emonet, R{\'e}mi and Fromont, Elisa and Habrard, Amaury and METZLER, Guillaume and Sebban, Marc},
  BOOKTITLE = {{AISTATS 2019 - 22nd International Conference on Artificial Intelligence and Statistics}},
  ADDRESS = {Naha, Okinawa, Japan},
  SERIES = {The 22nd International Conference on Artificial Intelligence and Statistics},
  VOLUME = {89},
  NUMBER = {1},
  PAGES = {1245-1253},
  YEAR = {2019},
}
```

### Code usage
```
python3 experiment.py [-h] --dataset DATASET --log_dir LOG_DIR --algo ALGO
                      [--C_grid C [C ...]] [--fold_grid FOLD [FOLD ...]]
                      [--max_step MAX_STEP] [--classif CLASSIF]
                      [--save_predictions] [--tmin TMIN] [--tmax TMAX]
                      [--beta BETA] [--strategy STRATEGY] [--save_states]
                      [--cone_with_state] [--state_size STATE_SIZE]
                      [--kappa KAPPA]
                      [--depth_grid DEPTH_GRID [DEPTH_GRID ...]]
                      [--nb_features NB_FEATURES]
```

#### arguments:
  `--dataset DATASET`     Dataset file
  `--log_dir LOG_DIR`     Logs directory
  `--algo ALGO`           (cone|parambath|baseline|ir)


  `-h, --help`            show help message and exit
  `--C_grid C [C ...]`    Grid over values of C
                        `default: [2^{-6}, 2^{-5}, ..., 2^{6}]`

  `--fold_grid FOLD [FOLD ...]`   Tested folds
                                `default: [0, 1, 2, 3, 4]`

  `--max_step MAX_STEP`   Maximum number of trained classifiers
                        `default: 19`

  `--classif CLASSIF`     Classifier (logi_reg|linear_svm|SVC_(linear|poly|rbf|sigmoid)|random_forest)
                        `default: linear_svm`

  `--save_predictions`    Enable saving all predictions
                        (required for thresholding, warning: results from large dataset can be heavy)

  `--tmin TMIN`           if algo is cone or parambath: inf bound of t space
                       `default: 0.0`

  `--tmax TMAX`           if algo is cone or parambath: sup bound of t space
                        `default: 1.0`

  `--beta BETA`           if algo is cone or parambath: beta used in class weight computation
                        `default: 1.0`

  `--strategy STRATEGY`   if algo is cone: strategy for cone selection (left|right|middle_space|middle_cones)
                        `default: middle_cones`

  `--save_states`         if algo is cone: Enable saving of all the intermediate states as png

  `--cone_with_state`     if algo is cone: Enable use of implementation with discretised (t, fm) space (slower implementation)

  `--state_size STATE_SIZE`   if algo is cone: size of the discretised (t, fm) space
                            `default: 10000`

  `--kappa KAPPA`         if algo is bisection: number of tuning iterations
                        `default: 100`

  `--depth_grid DEPTH_GRID [DEPTH_GRID ...]`   random forest max depth (replace C grid)
                                             `default: [2^{0}, 2^{1}, ..., 2^{6}]`

  `--nb_features NB_FEATURES`   if classif is DCA_ERSVM or H_ERSVM: number of features in dataset
                              `default: 0`
