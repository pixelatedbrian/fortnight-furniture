import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


# v1.04 load from stage3_imgs which is augmented by all being flipped

# make a function so the dataframe apply isn't super messy
def path_to_label(path):
    '''
    Take in a path like:
    ../data/stage1_imgs/5766_86.jpg
    and extract the label (86)

    Returns:
    int label: 86
    '''
    temp = path.split("/")[-1].split(".")[0].split("_")[-1]
    temp = int(temp)
    return temp


def get_skfold_data(path="../data/stage3_imgs/*.jpg"):
    '''
    * Loads all files in hard coded directory
    * Gets the labels from the file names
    * Splits into 10 StratifiedKFold
    * Builds a dictionary of fold indicies

    Returns:
    dictionary of fold indices like:
        {train_1:[indicies_1], test_1:[indicies_1], ..., test_10:[indicies_10] }
    '''
    image_paths = glob.glob(path)

    # move list into pandas so we can manipulate it easier
    data = pd.DataFrame(image_paths, columns=["file_path"])

    # data.head()

    data.loc[:, "y"] = data.loc[:, "file_path"].apply(path_to_label)

    # instantiate s-kfold object and prep splits
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf.get_n_splits(data.loc[:, "file_path"], data.loc[:, "y"])

    index_dict = {}     # this will hold the indices for each fold
    # so it will look like:
    # {train_1:[indicies_1], test_1:[indicies_1], ..., test_10:[indicies_10] }

    idx = 0   # keep track of which split we're on
    for train_index, test_index in skf.split(data.loc[:, "file_path"], data.loc[:, "y"]):
        idx += 1

        # get the train paths and matching labels
        index_dict["X_train_" + str(idx)] = data.loc[:, "file_path"].values[train_index]
        index_dict["y_train_" + str(idx)] = data.loc[:, "y"].values[train_index]

        # get the test paths and matching labels
        index_dict["X_test_" + str(idx)] = data.loc[:, "file_path"].values[test_index]
        index_dict["y_test_" + str(idx)] = data.loc[:, "y"].values[test_index]

    return index_dict
