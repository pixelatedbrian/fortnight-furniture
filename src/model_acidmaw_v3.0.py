# from keras.applications import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import load_img
from deepcnn import NewResNet50

from keras.optimizers import Adam, SGD
from keras import regularizers

# Keras imports
# from keras.models import  Sequential, Model
# from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.callbacks import EarlyStopping


# import glob
import json
# import numpy as np

from loader_bot_omega import LoaderBot   # dynamic full image augmentation
# from loader_bot import LoaderBot

import time
# from splitter import get_skfold_data
from plot_model import plot_hist
# import pandas as pd
# import numpy as np

# import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# https://github.com/DeepLearningSandbox/DeepLearningSandbox/blob/master/transfer_learning/fine-tune.py


def setup_to_finetune(model, freeze, optimizer, weight_decay):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
    model: keras model
    freeze: number of layers to keep frozen
    optimizer: which optimizer to use so the model can be compiled
    weight_decay: how much to regularize the thawed layers
    """
    for layer in model.layers[:freeze]:
        layer.trainable = False

    for layer in model.layers[freeze:]:
        layer.trainable = True
        # regularize unfrozen layers (new as of model v2.5)
        # https://github.com/keras-team/keras/issues/2717
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = regularizers.l2(weight_decay)
            # print("        adding regularization to thawed layer")

    # # regularize all layers:
    # for layer in model.layers:
    #     if hasattr(layer, 'kernel_regularizer'):
    #         layer.kernel_regularizer = regularizers.l2(WEIGHT_DECAY)

    # adam = Adam(lr=lr)
    # sgd = SGD(lr=lr, momentum=0.9)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


def temp_clean_history(hist):
    '''
    For whatever reason saving and loading weights prepends a 0 to the history
    of the model. (weird)

    This attempts to strip that padding so that the charts appear as they Should

    INPUTS:
    hist: dictionary with the history of a model (acc, val, etc)

    RETURNS:
    dictionary of lists with the padded zeros removed
    '''
    temp_hist = hist.copy()
    chop = sum([1 for item in hist["acc"] if item == 0])

    for key in temp_hist.keys():
        temp_hist[key] = temp_hist[key][chop:]

    return temp_hist


def run():
    # data_link_dict = get_skfold_data(path="../data/imgs/*.jpg")
    start_time = time.time()

    # decommisioned because inflight data augmentation solves a lot of these
    # problems

    # Use json to load the permanent dictionary that has been Created
    with open("../data/data_splits.json") as infile:
        data_link_dict = json.load(infile)

    EPOCHS = 10
    AUGMENTATION = 1    # could do 3 epochs of 10 augmentation or 30 of 1 which
                        # provides more data for plots to work with

    MINITRAINS = 5
    DO = 0.55  # drop out

    # for Adam inital LR of 0.0001 is a good starting point
    # for SGD initial LR of 0.001 is a good starting point
    LR = 0.0025
    # DECAY = 0.5e-6
    L2_REG = 0.05
    # OPTIMIZER = Adam(lr=LR, decay=DECAY)
    OPTIMIZER = SGD(lr=LR, momentum=0.9, nesterov=True)

    MODEL_ID = 'v2_5n'

    plot_file = "model_{:}.png".format(MODEL_ID)
    weights_file = "weights/model_{:}_weights.h5".format(MODEL_ID)
    history_file = "histories/history_{:}.json".format(MODEL_ID)

    # # user parameters for LoaderBot v1.0
    # # Parameters for Generators
    # params = {'dim': (299, 299),
    #           'batch_size': 256,
    #           'n_classes': 128,
    #           'n_channels': 3,
    #           'shuffle': False}

    # These parameters are for LoaderBot v2.0
    # Parameters for Generators
    params = {'dim': (299, 299),
              'batch_size': 64,
              'n_classes': 128,
              'n_channels': 3,
              'augmentation': AUGMENTATION,
              'shuffle': True}
    #
    # Parameters for Generators
    test_params = {'dim': (299, 299),
                   'batch_size': 64,
                   'n_classes': 128,
                   'n_channels': 3,
                   'augmentation': 1,
                   'augment': False,
                   'shuffle': False}

    # Datasets
    X_train_img_paths = data_link_dict["X_test_1"]
    y_train = data_link_dict["y_test_1"]

    X_test_img_paths = data_link_dict["X_test_2"]
    y_test = data_link_dict["y_test_2"]

    # Generators
    training_generator = LoaderBot(X_train_img_paths, y_train, **params)
    validation_generator = LoaderBot(X_test_img_paths, y_test, **test_params)

    # setup model
    # base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
    # model = regular_brian_layers(base_model, 128, DO, l1_reg=0.0005)

    model = NewResNet50()

    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

    # print(model.summary())

    # mini-train 1, like normal
    # transfer learning
    # setup_to_transfer_learn(model, base_model, OPTIMIZER)

    history = {}   # preinitialize history log

    for mt in range(MINITRAINS):
        temp = mt + 1

        if temp == 1:
            # Run model
            new_history = model.fit_generator(generator=training_generator,
                                             validation_data=validation_generator,
                                             epochs=EPOCHS,
                                             use_multiprocessing=False)

            history["acc"] = new_history.history["acc"]
            history["val_acc"] = new_history.history["val_acc"]
            history["loss"] = new_history.history["loss"]
            history["val_loss"] = new_history.history["val_loss"]

        else:

            temp_lr = LR / (10.0**(mt / 5 - (mt // 7.5)))
            print("\n\nLearning rate for mini-train: {:2.8f}\n\n".format(temp_lr))
            # mini-train 2
            OPTIMIZER = Adam(lr=temp_lr, decay=0.0)
            # try to fine tune some of the InceptionV3 layers also

            model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

            print("\n\n        Starting epoch {:}\n\n".format(EPOCHS * mt + 1))

            # Run model
            new_history = model.fit_generator(generator=training_generator,
                                              validation_data=validation_generator,
                                              epochs=EPOCHS,
                                              use_multiprocessing=False)

            # save the weights in case we want to predict on them later
            model.save(weights_file)

            history["acc"] += new_history.history["acc"]
            history["val_acc"] += new_history.history["val_acc"]
            history["loss"] += new_history.history["loss"]
            history["val_loss"] += new_history.history["val_loss"]

        # seems to be prepending a 0 to the list so ignore that
        history["acc"] = history["acc"]
        history["val_acc"] = history["val_acc"]
        history["loss"] = history["loss"]
        history["val_loss"] = history["val_loss"]

        temp_hist = temp_clean_history(history)

        plot_hist(temp_hist, plot_file, epochs=len(history["acc"]), sprint=True)

    # try to save the history so models can be more easily compared and Also
    # to better log results if going back is needed
    with open(history_file, "w") as outfile:
        json.dump(history, outfile)

    print("\n\n\n\nCompleted in {:6.2f} hrs".format(((time.time() - start_time)) / 3600))  # convert to hours


if __name__ == "__main__":
    run()
