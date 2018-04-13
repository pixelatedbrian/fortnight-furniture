import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


# v1.05 for splitting purposes only load non-augmented images, then after train
#   set is determined use that list to find augmented files of the same root file
#   thus avoiding using augmented images in the validation sets

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
    all_image_paths = glob.glob(path)

    # move list into pandas so we can manipulate it easier
    data = pd.DataFrame(all_image_paths, columns=["file_path"])

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
        # index_dict["X_test_" + str(idx)] = data.loc[:, "file_path"].values[test_index]
        # index_dict["y_test_" + str(idx)] = data.loc[:, "y"].values[test_index]

        temp_x = list(data.loc[:, "file_path"].values[test_index])
        temp_y = list(data.loc[:, "y"].values[test_index])

        # actually just most efficient/easiest to remove the superfluous y_labels
        # from the test sets

        # going to have to keep referencing these names so just make temp vars
        x_t = "X_test_" + str(idx)
        y_t = "y_test_" + str(idx)

        # Len BEFORE
        print("splitter.py -> get_skfold_data -> Before len(X_test_{:}) = {:}".format(idx, len(temp_x)))

        end = len(temp_x)
        idy = 0  # basically like for loop
        brakes = 0  # keep while loop from running forever
        while idy < end:
            brakes += 1  # again keep it from running away

            if "flip" in temp_x[idy]:
                temp_x.pop(idy)    # remove from X_train data
                temp_y.pop(idy)    # and also from the y_labels

                # since we popped something from a list update idy and end
                # since the list index information will have changed
                idy -= 1
                end -= 1

            idy += 1    # increment idy sort of like for loop

            if brakes > 1000000:
                print("splitter.py -> get_skfold_data breaking while loop")
                break

        # Len AFTER
        print("splitter.py -> get_skfold_data -> AFTER len(X_test_{:}) = {:}".format(idx, len(temp_x)))

        index_dict["X_test_" + str(idx)] = np.array(temp_x)
        index_dict["y_test_" + str(idx)] = np.array(temp_y)

    return index_dict


def get_skfold_data2(path="../data/stage3_imgs/*.jpg"):
    '''
    * Loads all files in hard coded directory
    * Gets the labels from the file names
    * Splits into 10 StratifiedKFold
    * Builds a dictionary of fold indicies

    Trying to fix augmented results being added to test set. First attempt was
    efficient but also resulted in slightly unbalanced test sets which would
    get more unbalanced as augmentation increased.

    So need to look at the original clean list of files, split, then pull in the
    augmented file paths to the training sets.

    Returns:
    dictionary of fold indices like:
        {train_1:[indicies_1], test_1:[indicies_1], ..., test_10:[indicies_10] }
    '''
    all_image_paths = glob.glob(path)

    # strip augmented files from the path list so we can do a clean strat-kfold split
    image_paths = [path for path in all_image_paths if "flip" not in path]

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
        # index_dict["X_train_" + str(idx)] = data.loc[:, "file_path"].values[train_index]
        # index_dict["y_train_" + str(idx)] = data.loc[:, "y"].values[train_index]

        # get the test paths and matching labels
        index_dict["X_test_" + str(idx)] = data.loc[:, "file_path"].values[test_index]
        index_dict["y_test_" + str(idx)] = data.loc[:, "y"].values[test_index]

        # take the list of all files, drop the test ones, the remainder will be

        temp_x = list(data.loc[:, "file_path"].values[train_index])
        temp_y = list(data.loc[:, "y"].values[train_index])

        # actually just most efficient/easiest to remove the superfluous y_labels
        # from the test sets

        print(temp_x[0])

    #     # Len BEFORE
    #     print("splitter.py -> get_skfold_data -> Before len(X_train_{:}) = {:}".format(idx, len(temp_x)))
    #
    #     end = len(temp_x)
    #     idy = 0  # basically like for loop
    #     brakes = 0  # keep while loop from running forever
    #     while idy < end:
    #         brakes += 1  # again keep it from running away
    #
    #         if "flip" in temp_x[idy]:
    #             temp_x.pop(idy)    # remove from X_train data
    #             temp_y.pop(idy)    # and also from the y_labels
    #
    #             # since we popped something from a list update idy and end
    #             # since the list index information will have changed
    #             idy -= 1
    #             end -= 1
    #
    #         idy += 1    # increment idy sort of like for loop
    #
    #         if brakes > 1000000:
    #             print("splitter.py -> get_skfold_data breaking while loop")
    #             break
    #
    #     # Len AFTER
    #     print("splitter.py -> get_skfold_data -> AFTER len(X_train_{:}) = {:}".format(idx, len(temp_x)))
    #
    #     index_dict["X_train_" + str(idx)] = np.array(temp_x)
    #     index_dict["y_train_" + str(idx)] = np.array(temp_y)
    #
    # return index_dict


if __name__ == "__main__":
    get_skfold_data2()
