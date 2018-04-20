import numpy as np
import keras
import cv2


class LoaderBot(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(299, 299), n_channels=3,
                 n_classes=128, shuffle=False):
        '''
        Initialization
        Already shuffled by get_skfold_indicies so no need to shuffle again
        '''
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load(ID)

            # ID:
            # ../data/stage1_imgs/flip_116297_85.jpg

            # For some reason CV2 loads as BGR instead of RGB
            temp = cv2.imread(ID)

            b,g,r = cv2.split(temp)         # get b,g,r
            rgb_img = cv2.merge([r,g,b])    # switch it to rgb

            X[i,] = rgb_img

            # Store class
            # y[i] = self.labels[ID]
            y[i] = int(ID.split("/")[-1].split(".")[0].split("_")[-1]) - 1

        # Inceptionv3 was trained on images that were processed so that color
        # values varied between [-1, 1] therefore we need to do the same:

        X /= 255.0
        X -= 0.5
        X *= 2.0

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


class FireBot(keras.utils.Sequence):
    'Generates data for Model Predictions, for evaluation'
    def __init__(self, list_IDs, batch_size=32, dim=(299, 299), n_channels=3,
                 n_classes=128, shuffle=False):
        '''
        Initialization
        Already shuffled by get_skfold_indicies so no need to shuffle again
        '''
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.len = None
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        'Denotes the number of batches per epoch'
        # add one to the len to account for leftovers, then on getitem check
        # to see if it's the last batch
        self.len = int(np.floor(len(self.list_IDs) / self.batch_size)) + 1
        return self.len

    def __getitem__(self, index):
        'Generate one batch of data'

        # make sure that it's not the last batch:
        if index < self.len - 1:
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        else:   # it's the last batch, return all leftovers
            # self.indexes = np.arange(len(self.list_IDs))
            indexes = self.indexes[index * self.batch_size:]
            # print("        >>>>>>>>    Firebot start last set at index:", index * self.batch_size)
            # print("        >>>>>>>>    Firebot self.indexes:", self.indexes[4960:])
            # print("        >>>>>>>>    Firebot current index:", index)
            # print("        >>>>>>>>    Firebot max index = ", self.len)
            # print("        >>>>>>>>    Firebot indexes:", indexes)

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(list_IDs_temp), *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load(ID)

            # ID:
            # ../data/stage1_imgs/flip_116297_85.jpg

            # For some reason CV2 loads as BGR instead of RGB
            temp = cv2.imread(ID)

            b, g, r = cv2.split(temp)         # get b,g,r
            rgb_img = cv2.merge([r, g, b])    # switch it to rgb

            X[i, ] = rgb_img

            # Store class
            # y[i] = self.labels[ID]
            # y[i] = int(ID.split("/")[-1].split(".")[0].split("_")[-1]) - 1

        # Inceptionv3 was trained on images that were processed so that color
        # values varied between [-1, 1] therefore we need to do the same:

        X /= 255.0
        X -= 0.5
        X *= 2.0

        return X
