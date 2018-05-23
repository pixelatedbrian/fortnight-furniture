from keras.applications import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import load_img

from keras.optimizers import Adam, SGD

# Keras imports
from keras.models import  Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D, LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.callbacks import EarlyStopping
from keras import regularizers

# import glob
import json
import numpy as np
import pandas as pd

from loader_bot_hawk import LoaderBot   # static load image augmentation
from plot_model import plot_hist, clean_history

import time

# https://github.com/DeepLearningSandbox/DeepLearningSandbox/blob/master/transfer_learning/fine-tune.py


def setup_to_transfer_learn(model, base_model, optimizer):
    """Freeze all layers and compile the model"""

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


def regular_brian_layers(base_model, num_classes, dropout=0.2, l1_reg=0.01):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l1(l1_reg))(x) #new FC layer, random init
    # x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(512,
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l1(l1_reg))(x) #new FC layer, random init
    # x = Dense(512, activation='relu')(x) #new FC layer, random init
    x = Dropout(dropout)(x)

    x = Dense(256,
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l1(l1_reg))(x) #new FC layer, random init    # x = Dense(256, activation='relu')(x) #new FC layer, random init
    x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x) #new softmax layer

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def neo_brian_layers(base_model, num_classes, dropout=0.5):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048)(x)  # new FC layer, random init
    x = LeakyReLU()(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(1024)(x)  # new FC layer, random init
    x = LeakyReLU()(x)
    # x = Dense(512, activation='relu')(x) #new FC layer, random init
    x = Dropout(dropout)(x)

    x = Dense(512)(x)  # new FC layer, random init
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    x = Dense(256)(x)  # ew FC layer, random init
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x)  # new softmax layer

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


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


def activate_regularization(model):
    '''
    Adding regularization to pretrained model doesn't work unless you save the
    model and then reload it. At least according to:
    https://github.com/keras-team/keras/issues/9106

    So this function will save a model then reload it.

    INPUT:
    model: model that will be reloaded

    RETURNS:
    reloaded model
    '''
    print("Activate Regularization: Save Model...")
    model.save_weights("temp.h5")
    print("Activate Regularization: Save Model Complete.")

    print("Activate Regularization: Reload Model Weights...")
    model.load_weights("temp.h5")
    print("Activate Regularization: Reload Model Weights Complete.")

    return model


def get_data(path):
    '''
    Load in JSON data dictionary at path and return it as a data dictionary

    Inputs:
    path - where to load the JSON File

    Returns:
    dictionary with 4 lists inside of it: X_train, y_train, X_test, y_test
    '''
    # Use json to load the permanent dictionary that has been Created
    with open(path) as infile:
        data_link_dict = json.load(infile)

    # TRAIN
    X_train = data_link_dict["X_test_1"]
    # convert to dataframe with specific order so we know path is in index 0
    df = pd.DataFrame(X_train, columns=['path', 'aspect_ratio',
                                        'b_accum_err', 'b_mean', 'file_size',
                                        'g_accum_err', 'g_mean', 'h', 'pix_pb',
                                        'pixels', 'r_accum_err', 'r_mean', 'w'])
    X_train = df.values[:, 0]   # pull the paths from the dataframe, not using the rest

    y_train = data_link_dict["y_test_1"]

    # TEST
    X_test = data_link_dict["X_test_2"]
    # convert to dataframe with specific order so we know path is in index 0
    df = pd.DataFrame(X_test, columns=['path', 'aspect_ratio',
                                       'b_accum_err', 'b_mean', 'file_size',
                                       'g_accum_err', 'g_mean', 'h', 'pix_pb',
                                       'pixels', 'r_accum_err', 'r_mean', 'w'])
    X_test = df.values[:, 0]

    y_test = data_link_dict["y_test_2"]

    data_dict = {'X_train': X_train, 'y_train': y_train,
                 'X_test': X_test, 'y_test': y_test}

    return data_dict


def run():
    # data_link_dict = get_skfold_data(path="../data/imgs/*.jpg")
    start_time = time.time()

    EPOCHS = 5
    MINITRAINS = 30
    DO = 0.625  # drop out
    SPRINT = True   # is this run a full train or a sprint

    # for Adam inital LR of 0.0001 is a good starting point
    # for SGD initial LR of 0.001 is a good starting point
    LR = 0.00025
    DECAY = 0.5e-6
    L2_REG = 0.125
    OPTIMIZER = Adam(lr=LR, decay=DECAY)
    # OPTIMIZER = SGD(lr=LR, momentum=0.9, nesterov=True)

    NB_IV3_LAYERS_TO_FREEZE = 172
    MODEL_ID = 'v2_7l'

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
    params = {'batch_size': 32,
              'augment': True,
              'augment_odds': 1.0,
              'shuffle': True}
    #
    # Parameters for Generators
    test_params = {'batch_size': 32,
                   'augment': False,
                   'shuffle': False}

    # Datasets
    data = get_data("../data/metadata_splits.json")

    print(data["X_train"][:5])

    # Generators
    training_generator = LoaderBot(data["X_train"], data["y_train"], **params)
    validation_generator = LoaderBot(data["X_test"], data["y_test"], **test_params)

    # setup model
    base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
    model = neo_brian_layers(base_model, 128, DO)

    # print(model.summary())
    # model.load_weights("weights/model_v2_7g_weights.h5")

    # mini-train 1, like normal
    # transfer learning
    setup_to_transfer_learn(model, base_model, OPTIMIZER)

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

            # params["augment_odds"] += 0.05
            # if params["augment_odds"] > 1.0:
            #     params["augment_odds"] = 1.0
            # training_generator = LoaderBot(data["X_train"], data["y_train"], **params)

            temp_lr = LR / (10.0**(mt / 5 - (mt // 7.5)))
            # temp_lr = LR / 1.5**temp
            # mini-train 2
            OPTIMIZER = Adam(lr=temp_lr, decay=0.0)
            # try to fine tune some of the InceptionV3 layers also

            thaw_count = int(2.5 * temp)
            if thaw_count > 10:
                thaw_count = 10

            thaw_count = NB_IV3_LAYERS_TO_FREEZE - thaw_count

            setup_to_finetune(model, thaw_count, OPTIMIZER, L2_REG)

            model = activate_regularization(model)

            print("\n\n        Starting epoch {:}\n\n".format(EPOCHS * mt + 1))
            print("Learning rate for mini-train: {:2.8f}   augmentation odds:{:2.2f}\n\n".format(temp_lr, params["augment_odds"]))

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

        plot_hist(history, plot_file, epochs=len(history["acc"]), sprint=SPRINT)

    # try to save the history so models can be more easily compared and Also
    # to better log results if going back is needed
    with open(history_file, "w") as outfile:
        history = clean_history(history)  # clean up zero padding, if it exists
        json.dump(history, outfile)

    print("\n\n\n\nCompleted in {:6.2f} hrs".format(((time.time() - start_time)) / 3600))  # convert to hours


if __name__ == "__main__":
    run()
