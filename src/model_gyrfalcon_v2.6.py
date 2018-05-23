from keras.applications import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import load_img

from keras.optimizers import Adam, SGD

# Keras imports
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D, BatchNormalization
# from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.layers import concatenate

# import glob
import json
import numpy as np
import pandas as pd

from loader_bot_reloaded import LoaderBot  # include meta features
# from loader_bot_omega import LoaderBot   # dynamic full image augmentation
# from loader_bot import LoaderBot

import time
from plot_model import plot_hist
# import pandas as pd
# import numpy as np

# https://github.com/DeepLearningSandbox/DeepLearningSandbox/blob/master/transfer_learning/fine-tune.py


def setup_to_transfer_learn(model, inception, raven, optimizer):
    """Freeze all layers and compile the model"""

    # remove softmax from raven_model
    raven.layers.pop()

    for _layer in raven.layers:
        _layer.trainable = False

    for _layer in inception.layers:
        _layer.trainable = False

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


def regular_meta_brian_layers(base_model, num_classes, dropout=0.2, l1_reg=0.01):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    ##############################################
    ### BUILD MINI-DNN FOR IMAGE META-FEATURES ###
    ##############################################
    meta_features_x = Input(shape=(12,))

    mx = Dense(128)(meta_features_x)
    mx = BatchNormalization()(mx)
    mx = Dropout(0.15)(mx)
    mx = Activation("relu")(mx)

    mx = Dense(256)(meta_features_x)
    mx = BatchNormalization()(mx)
    mx = Dropout(0.15)(mx)
    mx = Activation("relu")(mx)

    mx = Dense(128)(meta_features_x)
    mx = BatchNormalization()(mx)
    mx = Dropout(0.15)(mx)
    mx = Activation("relu")(mx)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = concatenate([x, mx], axis=-1)

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

    model = Model(inputs=[base_model.input, meta_features_x], outputs=predictions)

    return model


def meta_brian_layers(base_model, raven_model, num_classes,
                      dropout=0.2, l1_reg=0.01):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    mx = raven_model.output

    x = concatenate([x, mx], axis=-1)

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

    model = Model(inputs=[base_model.input, raven_model.input], outputs=predictions)

    return model


def add_brian_layers(base_model, num_classes, dropout=0.2):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x) #new FC layer, random init
    # x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(512, activation='relu', kernel_initializer='he_normal')(x) #new FC layer, random init
    # x = Dense(512, activation='relu')(x) #new FC layer, random init
    x = Dropout(dropout)(x)

    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x) #new FC layer, random init
    # x = Dense(256, activation='relu')(x) #new FC layer, random init
    x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x) #new softmax layer

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def thaw_raven(raven_model, optimizer, weight_decay):
    '''
    Thaw all of the Raven DNN layers so that it can resume learning

    Args:
    raven_model: keras DNN
    optimizer:  which optimizer to use when recompiling the Model
    weight_decay: how much to regularize the thawed layers
    '''
    for layer in raven_model.layers:
        layer.trainable = True

        # regularize unfrozen layers (new as of model v2.5)
        # https://github.com/keras-team/keras/issues/2717
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = regularizers.l2(weight_decay)


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


def Raven(DO=0.25, classes=128, load_weights=False):
    _input = Input(shape=(12,))

    # First Layer
    x = Dense(128)(_input)
    x = BatchNormalization()(x)
    x = Dropout(DO)(x)
    x = Activation("relu")(x)

    # Second Layer
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Dropout(DO)(x)
    x = Activation("relu")(x)

    # Third Layer
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Dropout(DO)(x)
    x = Activation("relu")(x)

    output = Dense(classes, activation='softmax')(x)

    model = Model(_input, output, name='Raven_0.1')

    if load_weights is True:
        model.load_weights("weights/err_0.091_model_raven.h5")

    return model


def run():
    # data_link_dict = get_skfold_data(path="../data/imgs/*.jpg")
    start_time = time.time()

    # decommisioned because inflight data augmentation solves a lot of these
    # problems

    # Use json to load the permanent dictionary that has been Created
    with open("../data/metadata_splits.json") as infile:
        data_link_dict = json.load(infile)

    EPOCHS = 5

    MINITRAINS = 20  # with 20 epoch for initial train it will add to 100 total
    DO = 0.55  # drop out

    # for Adam inital LR of 0.0001 is a good starting point
    # for SGD initial LR of 0.001 is a good starting point
    LR = 0.00025
    DECAY = 0.5e-6
    L2_REG = 0.025
    OPTIMIZER = Adam(lr=LR, decay=DECAY)
    # OPTIMIZER = SGD(lr=LR, momentum=0.9, nesterov=True)

    NB_IV3_LAYERS_TO_FREEZE = 172
    MODEL_ID = 'v2_6k'

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

    R_PICS = 1

    # These parameters are for LoaderBot v2.0
    # Parameters for Generators
    params = {'batch_size': 32,
              'random_pics': R_PICS,
              'augment': False,
              'percent_random': 0.0,
              'shuffle': True}
    #
    # Parameters for Generators
    test_params = {'batch_size': 32,
                   'percent_random': 0.0,
                   'random_pics': R_PICS,
                   'shuffle': False}

    # Datasets

    # get list of dictionaries from the data dictionary
    X_train = data_link_dict["X_test_1"]

    # convert to dataframe with specific order so we know path is in index 0
    df = pd.DataFrame(X_train, columns=['path', 'aspect_ratio',
                                        'b_accum_err', 'b_mean', 'file_size',
                                        'g_accum_err', 'g_mean', 'h', 'pix_pb',
                                        'pixels', 'r_accum_err', 'r_mean', 'w'])
    X_train = df.values

    # X_train_img_paths = data_link_dict["X_test_1"]
    y_train = data_link_dict["y_test_1"]

    X_test = data_link_dict["X_test_2"]

    # convert to dataframe with specific order so we know path is in index 0
    df = pd.DataFrame(X_test, columns=['path', 'aspect_ratio',
                                       'b_accum_err', 'b_mean', 'file_size',
                                       'g_accum_err', 'g_mean', 'h', 'pix_pb',
                                       'pixels', 'r_accum_err', 'r_mean', 'w'])
    X_test = df.values

    # X_test_img_paths = data_link_dict["X_test_2"]
    y_test = data_link_dict["y_test_2"]

    print("length of X_train_img_paths:", len(X_train))

    del data_link_dict

    # Generators
    training_generator = LoaderBot(X_train, y_train, **params)
    validation_generator = LoaderBot(X_test, y_test, **test_params)

    # setup model
    inception = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer

    # DNN for meta-image features with optional transfer learning
    raven_model = Raven(0.05, classes=128, load_weights=True)

    model = meta_brian_layers(inception, raven_model, 128, DO, l1_reg=0.0001)
    # model = regular_brian_layers(base_model, 128, DO, l1_reg=0.0005)

    # print(model.summary())

    # model.load_weights("weights/model_v2_6h_weights.h5")

    # mini-train 1, like normal
    # transfer learning
    setup_to_transfer_learn(model, inception, raven_model, OPTIMIZER)

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
            # temp_lr = LR / 1.5**mt
            temp_lr = LR / (10.0**(mt / 5 - (mt // 7.5)))
            print("\n\nLearning rate for mini-train: {:2.8f}\n\n".format(temp_lr))
            # mini-train 2
            OPTIMIZER = Adam(lr=temp_lr, decay=0.0)
            # try to fine tune some of the InceptionV3 layers also

            thaw_count = int(2.5 * temp)
            if thaw_count > 50:
                thaw_count = 50

            thaw_count = NB_IV3_LAYERS_TO_FREEZE - thaw_count

            if temp > 5:
                thaw_raven(raven_model, OPTIMIZER, L2_REG)

            setup_to_finetune(model, thaw_count, OPTIMIZER, L2_REG)

            model = activate_regularization(model)

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

        plot_hist(history, plot_file, epochs=len(history["acc"]), sprint=False)

        # try to save the history so models can be more easily compared and Also
        # to better log results if going back is needed
        with open(history_file, "w") as outfile:
            json.dump(history, outfile)

    print("\n\n\n\nCompleted in {:6.2f} hrs".format(((time.time() - start_time)) / 3600))  # convert to hours


if __name__ == "__main__":
    run()
