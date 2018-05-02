import numpy as np
import keras
import cv2
from image_aug import fancy_pca, rotate_bound, smart_crop, crop_image


class LoaderBot(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(299, 299), n_channels=3,
                 n_classes=128, shuffle=False, augmentation=1, augment=True):
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
        self.len = None
        self.augmentation = augmentation
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        self.len = int(np.floor(len(self.list_IDs) / self.batch_size)) * self.augmentation
        return self.len

    def __getitem__(self, index):
        'Generate one batch of data'

        # make sure that it's not the last batch:
        if index < self.len:
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        else:   # it's the last batch, return all leftovers
            # self.indexes = np.arange(len(self.list_IDs))
            indexes = self.indexes[index * self.batch_size:]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # print("len list_IDs_temp", len(list_IDs_temp))

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # since there is augmentation add the same items to a bigger list
        # as many extra times as augmentation is, ie 2x or 10x
        self.indexes = np.hstack([np.arange(len(self.list_IDs)) for i in range(self.augmentation)])

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

            # Store class
            # y[i] = self.labels[ID]
            y[i] = int(ID.split("/")[-1].split(".")[0].split("_")[-1]) - 1

            # do some error checking because some files suck
            # for example some files are shape(1, 1, 3)
            # if temp.shape[0] is 1 or temp.shape[1] is 1:
            #     print("shape", temp.shape)
            #     pass    # skip this one
            # else:

            b, g, r = cv2.split(temp)         # get b,g,r
            temp_img = cv2.merge([r, g, b])    # switch it to rgb

            # for normal training with augmentation
            if self.augment is True:

                # roll the dice on augmentation, 2 because not inclusive like lists
                _flip = np.random.randint(0, 2)

                pre_flip = temp_img.copy()

                if _flip is 1:
                    temp_img = temp_img[:, ::-1, :]  # flip image over vertical axis

                # rotate?
                _rot = np.random.randint(0, 2)
                # trying to measure how changes to augmentation... change things
                # _rot = 0

                pre_rot = temp_img.copy()

                if _rot is 1:
                    # figure out how much rotation (should be about -15 to 15 degrees
                    # but a normal distribution centered on 0)
                    _rotation = np.random.normal(0, 1.0)

                    # rotate and THEN also crop so there's not a lot of black on the
                    # edges
                    temp_img = rotate_bound(temp_img, _rotation)

                pre_crop = temp_img.copy()

                # get the sub-crop of the bigger image
                temp_img = smart_crop(temp_img)

                # do fancy PCA color augmentation
                temp_img = fancy_pca(temp_img, alpha_std=0.10)

            else:
                # just get the middle of the image, resized to 299^2
                temp_img = crop_image(temp_img)
                # print("ID:", ID, "Label:", y[i])

            try:
                # image is just a square but not necessarily 299px x 299px
                # generalize to self.dim so other 'nets can be used, like VGG16
                resized = cv2.resize(temp_img, self.dim, interpolation=cv2.INTER_AREA)
            except:
                print("something went horribly wrong in LoaderBot _generate_data", ID, "shape", temp.shape)
                print("pre_flip", pre_flip.shape)
                print("pre_rot", pre_rot.shape)
                print("pre_crop", pre_crop.shape)
                print("new size:", temp_img.shape)

            resized = resized / 255.0   # convert to float

            # remove imagenet means from values
            resized[..., 0] -= (103.939 / 255.0)    # red
            resized[..., 1] -= (116.779 / 255.0)    # green
            resized[..., 2] -= (123.68 / 255.0)     # blue

            X[i, ] = resized

        # Inceptionv3 was trained on images that were processed so that color
        # values varied between [-1, 1] therefore we need to do the same:

        # X /= 255.0
        # X -= 0.5   # can probably remove this since we subtracted the imagenet means
        # X *= 2.0

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
