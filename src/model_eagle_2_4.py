# from keras.applications import InceptionV3
# from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50    # keras stand-alone?
# from tensorflow.contrib.keras.applications import ResNet50

# import tensorflow.contrib.keras.applications.ResNet50

# REFERENCES:

# VGG16
# https://arxiv.org/abs/1409.1556

from keras.optimizers import Adam, SGD

# Keras imports
from keras.models import  Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.callbacks import EarlyStopping

# import glob
import json

from loader_bot_omega import LoaderBot   # dynamic full image augmentation
# from loader_bot import LoaderBot

import time
from splitter import get_skfold_data

# import pandas as pd
# import numpy as np

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# https://github.com/DeepLearningSandbox/DeepLearningSandbox/blob/master/transfer_learning/fine-tune.py

def setup_to_transfer_learn(model, base_model, optimizer):
    """Freeze all layers and compile the model"""

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


def add_brian_layers(base_model, num_classes, dropout=0.5):
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


def add_double_brian_layers(base_model, num_classes, dropout=0.5):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x) #new FC layer, random init
    # x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x) #new FC layer, random init
    # x = Dense(512, activation='relu')(x) #new FC layer, random init
    x = Dropout(dropout)(x)

    x = Dense(512, activation='relu', kernel_initializer='he_normal')(x) #new FC layer, random init
    # x = Dense(256, activation='relu')(x) #new FC layer, random init
    x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x) #new softmax layer

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def setup_to_finetune(model, freeze, optimizer):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
    model: keras model
    """
    for layer in model.layers[:freeze]:
        layer.trainable = False

    for layer in model.layers[freeze:]:
        layer.trainable = True

    # adam = Adam(lr=lr)
    # sgd = SGD(lr=lr, momentum=0.9)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


def plot_hist(history, info_str, epochs=2, augmentation=1, sprint=False):
    '''
    Make a plot of the rate of error as well as the accuracy of the model
    during training.  Also include a line at error 0.20 which was the original
    minimum acceptable error (self imposed) to submit results to the test
    set when doing 3-way split.
    Even after performance regularly exceeded the minimum requirement the line
    was unchanged so that all of the graphs would be relative to each other.
    Also it was still useful to see how a model's error was performing relative
    to this baseline.
    Also, the 2 charts written as a png had the filename coded to include
    hyperparameters that were used in the model when the chart was created.
    This allowed a simple visual evaluation of a model's performance when
    doing randomized hyperparameter search. If a model appeared to be high
    performing then the values could be reused in order to attempt to
    replicate the result.
    '''
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    fig.suptitle("", fontsize=12, fontweight='normal')

    # stuff for marking the major and minor ticks dynamically relative
    # to the numper of epochs used to train
    major_ticks = int(epochs / 10.0)
    minor_ticks = int(epochs / 20.0)

    title_text = "Homewares and Furniture Image Identification\n Train Set and Dev Set"
    ACC = 0.817   # record accuracy
    if sprint is True:
        ACC = 0.740
        title_text = "SPRINT: Homewares and Furniture Image Identification\n Train Set and Dev Set"

    if major_ticks < 2:
        major_ticks = 2

    if minor_ticks < 1:
        minor_ticks = 1

    majorLocator = MultipleLocator(major_ticks)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(minor_ticks)

    # correct x axis
    history['loss'] = [0.0] + history['loss']
    history['val_loss'] = [0.0] + history['val_loss']
    history['acc'] = [0.0] + history['acc']
    history['val_acc'] = [0.0] + history['val_acc']

    x_line = [ACC] * (epochs + 1)  # this line is now for accuracy of test set

    # stuff for the loss chart
    axs[0].set_title(title_text)

    if augmentation > 1:
        axs[0].set_xlabel('Epochs\nAugmentation of {:3d}'.format(augmentation))
    else:
        axs[0].set_xlabel('Epochs')

    axs[0].set_xlim(1, epochs)
    axs[0].set_ylabel('Loss')
#     axs[0].set_ylim(0, 15)

    axs[0].plot(history['loss'], color="blue", linestyle="--", alpha=0.8, lw=1.0)
    axs[0].plot(history['val_loss'], color="blue", alpha=0.8, lw=1.0)
    axs[0].legend(['Training', 'Validation'])
    axs[0].xaxis.set_major_locator(majorLocator)
    axs[0].xaxis.set_major_formatter(majorFormatter)

    # for the minor ticks, use no labels; default NullFormatter
    axs[0].xaxis.set_minor_locator(minorLocator)

    # stuff for the accuracy chart
    axs[1].set_title(title_text)

    if augmentation > 1:
        axs[0].set_xlabel('Epochs\nAugmentation of {:3d}'.format(augmentation))
    else:
        axs[0].set_xlabel('Epochs')

    axs[1].set_xlim(1, epochs)
    axs[1].set_ylabel('Accuracy')
    axs[1].set_ylim(0.0, 1.0)
    axs[1].plot(x_line, color="red", alpha=0.3, lw=4.0)
    axs[1].plot(history['acc'], color="blue", linestyle="--", alpha=0.5, lw=1.0)
    axs[1].plot(history['val_acc'], color="blue", alpha=0.8, lw=1.0)
    axs[1].plot(x_line, color="red", linestyle="--", alpha=0.8, lw=1.0)
    axs[1].legend(['Record Accuracy ({:1.2f})'.format(ACC), 'Training', 'Validation'], loc='lower right')
    axs[1].xaxis.set_major_locator(majorLocator)
    axs[1].xaxis.set_major_formatter(majorFormatter)

    # for the minor ticks, use no labels; default NullFormatter
    axs[1].xaxis.set_minor_locator(minorLocator)

    plt.savefig("../imgs/" + info_str, facecolor='w', edgecolor='w', transparent=False)
    # plt.show()


def run():
    data_link_dict = get_skfold_data(path="../data/imgs/*.jpg")
    start_time = time.time()

    # decommisioned because inflight data augmentation solves a lot of these
    # problems

    # Use json to load the permanent dictionary that has been Created
    # with open("../data/data_splits.json") as infile:
    #     data_link_dict = json.load(infile)

    EPOCHS = 10
    AUGMENTATION = 1    # could do 3 epochs of 10 augmentation or 30 of 1 which
                        # provides more data for plots to work with

    DO = 0.55  # drop out

    # for Adam inital LR of 0.0001 is a good starting point
    # for SGD initial LR of 0.001 is a good starting point
    LR = 0.000025
    DECAY = 0.5e-6
    OPTIMIZER = Adam(lr=LR, decay=DECAY)
    # OPTIMIZER = SGD(lr=LR, momentum=0.9, nesterov=True)

    # NB_IV3_LAYERS_TO_FREEZE = 172
    NB_IV3_LAYERS_TO_FREEZE = 178
    MODEL_ID = 'v2_4c'

    plot_file = "model_{:}.png".format(MODEL_ID)
    weights_file = "weights/model_{:}_weights.h5".format(MODEL_ID)
    history_file = "histories/history_{:}.json".format(MODEL_ID)

    # # user parameters for LoaderBot v1.0
    # # Parameters for Generators
    # params = {'dim': (224, 224),
    #           'batch_size': 64,
    #           'n_classes': 128,
    #           'n_channels': 3,
    #           'shuffle': False}

    # These parameters are for LoaderBot v2.0
    # Parameters for Generators
    params = {'dim': (224, 224),
              'batch_size': 64,
              'augmentation': AUGMENTATION,
              'augment': False,
              'shuffle': True}

    # Parameters for Generators
    test_params = {'dim': (224, 224),
                   'batch_size': 64,
                   'augment': False}

    # Datasets
    X_train_img_paths = data_link_dict["X_test_2"]
    y_train = data_link_dict["y_test_2"]

    X_test_img_paths = data_link_dict["X_test_3"]
    y_test = data_link_dict["y_test_3"]

    # Generators
    training_generator = LoaderBot(X_train_img_paths, y_train, **params)
    validation_generator = LoaderBot(X_test_img_paths, y_test, **test_params)

    # setup model
    # base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
    base_model = ResNet50(include_top=False, weights='imagenet')

    # seems like in Keras not including the top will exclude the FC layers at the
    # top, not just the softmax categories
    # # try to pop some layers to get to the top 'maxpool' then rebuild from there
    # base_model.pop()
    # base_model.pop()
    # base_model.pop()
    #
    # base_model.summary()

    model = add_double_brian_layers(base_model, 128, DO)

    # mini-train 1, like normal
    # transfer learning
    setup_to_transfer_learn(model, base_model, OPTIMIZER)

    model.summary()

    # print("model layers:", model.layers)
    print("len model layers:", len(model.layers))

    # Run model
    history_t1 = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     epochs=EPOCHS,
                                     use_multiprocessing=False)

    # mini-train 2
    OPTIMIZER = Adam(lr=LR / 2.0, decay=DECAY)
    # try to fine tune some of the InceptionV3 layers also
    setup_to_finetune(model, NB_IV3_LAYERS_TO_FREEZE - 2, OPTIMIZER)

    print("\n\n        Starting epoch {:}\n\n".format(EPOCHS + 1))

    # Run model
    history_t2 = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     epochs=EPOCHS,
                                     use_multiprocessing=False)

    # mini-train 3
    OPTIMIZER = Adam(lr=LR / 4.0, decay=DECAY)
    # try to fine tune some of the InceptionV3 layers also
    setup_to_finetune(model, NB_IV3_LAYERS_TO_FREEZE - 4, OPTIMIZER)

    print("\n\n        Starting epoch {:}\n\n".format(EPOCHS * 2 + 1))

    # Run model
    history_t3 = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     epochs=EPOCHS,
                                     use_multiprocessing=False)

    # mini-train 4
    OPTIMIZER = Adam(lr=LR / 8.0, decay=DECAY)
    # try to fine tune some of the InceptionV3 layers also
    setup_to_finetune(model, NB_IV3_LAYERS_TO_FREEZE - 6, OPTIMIZER)

    print("\n\n        Starting epoch {:}\n\n".format(EPOCHS * 3 + 1))

    # Run model
    history_t4 = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     epochs=EPOCHS,
                                     use_multiprocessing=False)

    # save the weights in case we want to predict on them later
    model.save(weights_file)

    history_tl = history_t1.history
    history_tl["acc"] += history_t2.history["acc"]
    history_tl["val_acc"] += history_t2.history["val_acc"]
    history_tl["loss"] += history_t2.history["loss"]
    history_tl["val_loss"] += history_t2.history["val_loss"]
    #
    history_tl = history_t1.history
    history_tl["acc"] += history_t3.history["acc"]
    history_tl["val_acc"] += history_t3.history["val_acc"]
    history_tl["loss"] += history_t3.history["loss"]
    history_tl["val_loss"] += history_t3.history["val_loss"]

    history_tl = history_t1.history
    history_tl["acc"] += history_t4.history["acc"]
    history_tl["val_acc"] += history_t4.history["val_acc"]
    history_tl["loss"] += history_t4.history["loss"]
    history_tl["val_loss"] += history_t4.history["val_loss"]

    plot_hist(history_tl, plot_file, epochs=len(history_tl["acc"]), sprint=True)

    # try to save the history so models can be more easily compared and Also
    # to better log results if going back is needed
    with open(history_file, "w") as outfile:
        json.dump(history_tl, outfile)

    print("\n\n\n\nCompleted in {:6.2f} hrs".format(((time.time() - start_time)) / 3600))  # convert to hours


if __name__ == "__main__":
    run()
