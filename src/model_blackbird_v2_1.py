from keras.applications import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import load_img

from keras.optimizers import Adam, SGD

# Keras imports
from keras.models import  Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.callbacks import EarlyStopping

# import glob
import json

from loader_bot_omega import LoaderBot   # dynamic full image augmentation

import time
from splitter import get_skfold_data

# import pandas as pd
# import numpy as np

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# https://github.com/DeepLearningSandbox/DeepLearningSandbox/blob/master/transfer_learning/fine-tune.py

def setup_to_transfer_learn(model, base_model, lr=0.0001):
    """Freeze all layers and compile the model"""

    for layer in base_model.layers:
        layer.trainable = False

    adam = Adam(lr=lr)
    sgd = SGD(lr=lr, momentum=0.9)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


def add_brian_light_layers(base_model, num_classes, dropout=0.2):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(140, activation='relu')(x) #new FC layer, random init
    # x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x) #new softmax layer

    model = Model(inputs=base_model.input, outputs=predictions)

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


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init

    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def setup_to_finetune(model, freeze, lr=0.001):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
    model: keras model
    """
    for layer in model.layers[:freeze]:
        layer.trainable = False

    for layer in model.layers[freeze:]:
        layer.trainable = True

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])


def plot_hist(history, info_str, epochs=2, augmentation=1):
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

    ACC = 0.817   # record accuracy

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
    axs[0].set_title("Homewares and Furniture Image Identification\n Train Set and Dev Set")

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
    axs[1].set_title("Homewares and Furniture Image Identification\n Train Set and Dev Set")

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

    # # Use json to load the permanent dictionary that has been Created
    # with open("../data/data_splits.json") as infile:
    #     data_link_dict = json.load(infile)

    EPOCHS = 20
    AUGMENTATION = 1    # could do 3 epochs of 10 augmentation or 30 of 1 which
                        # provides more data for plots to work with
    LR = 0.0001
    NB_IV3_LAYERS_TO_FREEZE = 172

    # Parameters for Generators
    params = {'dim': (299,299),
              'batch_size': 256,
              'n_classes': 128,
              'n_channels': 3,
              'augmentation': AUGMENTATION,
              'shuffle': True}

    # Parameters for Generators
    test_params = {'dim': (299,299),
                   'batch_size': 256,
                   'n_classes': 128,
                   'n_channels': 3,
                   'augmentation': 1,
                   'augment': False,
                   'shuffle': True}

    # Datasets
    X_train_img_paths = data_link_dict["X_test_1"]
    y_train = data_link_dict["y_test_1"]

    X_test_img_paths = data_link_dict["X_test_2"]
    y_test = data_link_dict["y_test_2"]

    # Generators
    training_generator = LoaderBot(X_train_img_paths, y_train, **params)
    validation_generator = LoaderBot(X_test_img_paths, y_test, **test_params)

    # setup model
    base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
    model = add_brian_light_layers(base_model, 128, 0.55)

    # mini-train 1, like normal
    # transfer learning
    setup_to_transfer_learn(model, base_model, lr=LR)

    # Run model
    history_t1 = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     epochs=EPOCHS,
                                     use_multiprocessing=False)

    # mini-train 2
    # try to fine tune some of the InceptionV3 layers also
    setup_to_finetune(model, NB_IV3_LAYERS_TO_FREEZE - 3, lr=LR / 2.0)

    # Run model
    history_t2 = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     epochs=EPOCHS,
                                     use_multiprocessing=False)

    # mini-train 3
    # try to fine tune some of the InceptionV3 layers also
    setup_to_finetune(model, NB_IV3_LAYERS_TO_FREEZE - 6, lr=LR / 4.0)

    # Run model
    history_t3 = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     epochs=EPOCHS,
                                     use_multiprocessing=False)

    # mini-train 4
    # try to fine tune some of the InceptionV3 layers also
    setup_to_finetune(model, NB_IV3_LAYERS_TO_FREEZE - 9, lr=LR / 8.0)

    # Run model
    history_t4 = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     epochs=EPOCHS,
                                     use_multiprocessing=False)

    model.save("model_v2_1c_weights.h5")

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

    plot_hist(history_tl, "model_v2_1c.png", epochs=len(history_tl["acc"]))



    print("\n\n\n\nCompleted in {:6.2f} hrs".format(((time.time() - start_time)) / 3600))  # convert to hours

if __name__ == "__main__":
    run()
