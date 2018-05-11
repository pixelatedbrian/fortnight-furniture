import pandas as pd
import numpy as np

from keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K

from sklearn.model_selection import StratifiedKFold

import json


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
    model = Model(inputs=_input, outputs=output)

    if load_weights is True:
        model.load("/weights/model_raven.h5")

    return model


with open("../data/metadata_splits.json") as infile:
    data = json.load(infile)

X_train = pd.DataFrame(data["X_train_1"],
                       columns=['h', 'w', 'pixels', 'aspect_ratio', 'file_size',
                       'pix_pb', 'r_mean', 'g_mean', 'b_mean', 'r_accum_err',
                       'g_accum_err', 'b_accum_err'])
X_test = pd.DataFrame(data["X_test_1"],
                      columns=['h', 'w', 'pixels', 'aspect_ratio', 'file_size',
                      'pix_pb', 'r_mean', 'g_mean', 'b_mean', 'r_accum_err',
                      'g_accum_err', 'b_accum_err'])

y_train = np.array(data["y_train_1"])
y_test = np.array(data["y_test_1"])

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.25,
                                   patience=25,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='min')

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=50, verbose=1, mode='min')

callbacks = [reduce_lr_loss]

LR = 0.005
DECAY = 1e-5
DO = 0.10
EPOCHS = 500
# EPOCHS = 100
MINI_TRAINS = 15
BATCH = 2048

# X_train
# zero base y_train
y_train -= 1
y_train = to_categorical(y_train, 128)

# X_test
# zero base y_test
y_test -= 1
y_test = to_categorical(y_test, 128)


OPTIMIZER = Adam(lr=LR, decay=DECAY)

model = Raven(DO, classes=128, load_weights=False)

model.compile(optimizer=OPTIMIZER,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=EPOCHS, batch_size=BATCH, callbacks=callbacks,
          verbose=1)

test_score = model.evaluate(X_test, y_test, batch_size=BATCH, verbose=0)
train_score = model.evaluate(X_train, y_train, batch_size=BATCH, verbose=0)

# if idx == 0:
print("\n\n")

print("Train: {:2.2f}% Test : {:2.2f}%".format(train_score[1] * 100,
                                               test_score[1] * 100))
path = "weights/err_{:2.3f}_model_raven.h5".format(test_score[1])

print("path: ", path)

model.save(path)
