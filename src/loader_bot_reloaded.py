import numpy as np
import keras
import cv2
from image_aug import fancy_pca, rotate_bound, smart_crop, crop_image


# Reloaded is version 3 of the loaderbot class.  The main thing is being able
# to take in meta features from the images and concatenate them into the model
# as it dynamically loads data for the model


class LoaderBot(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_data, labels, batch_size=32, dim=(299, 299),
                 n_channels=3, n_classes=128, shuffle=False,
                 augment=False, random_pics=1, percent_random=0.1):
        '''
        Initialization
        Already shuffled by get_skfold_indicies so no need to shuffle again
        '''
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        # self.list_IDs = list_IDs
        self.x_data = x_data
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.len = len(self.x_data)
        self.augment = augment
        self.percent_random = percent_random
        self.random_pics = random_pics   # the number of rolling options for pics
                                         # for static augmentation, ie each pic
                                         # has 20 pre-made random augmentations
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        self.len = int(np.floor(len(self.x_data) / self.batch_size)) * self.augmentation
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
        x_data_slice = np.array([self.x_data[k] for k in indexes])

        # print("len list_IDs_temp", len(list_IDs_temp))

        # Generate data
        if self.random_pics > 1:
            # shoulder be faster but also limited
            X, y = self.new_data_generation(x_data_slice)
        else:
            # make augmented images on the fly (slow but broad)
            X, y = self.__data_generation(x_data_slice)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # since there is augmentation add the same items to a bigger list
        # as many extra times as augmentation is, ie 2x or 10x
        self.indexes = np.arange(len(self.x_data))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def new_data_generation(self, x_data_slice):
        '''
        Load random but static augmented images
        '''
        # Initialization
        X = np.empty((x_data_slice.shape[0], *self.dim, self.n_channels))
        # print("x_data_slice type:", type(x_data_slice))
        X_meta = x_data_slice[:, 1:]
        y = np.empty((x_data_slice.shape[0]), dtype=int)

        # generate data
        for idx, x_data in enumerate(x_data_slice):

            # see if you will select a random pic or just a normal pic

            roll = np.random.random()
            # file name to modify
            # _file = x_data[0].split("/")[-1]

            if roll > self.percent_random:
                # Don't pick an augmented pic

                temp_path = "../data/stage3_imgs/" + x_data[0].split("/")[-1]
                temp_img = cv2.imread(temp_path)

                if temp_img.shape[0] < 299 or temp_img.shape[1] < 299:
                    print("temp_img shape messed up in no random", temp_img.shape)
                # print("not_random temp_img.shape", temp_img.shape)
            else:
                # pick an augmented pic
                # print("pick a random pic")
                # random pic number
                pic = np.random.randint(0, self.random_pics)   # randint isn't inclusive

                # reconstruct path to load
                temp_path = "../data/static_aug/" + str(pic) + "_" + x_data[0].split("/")[-1]
                # new_path = "../data/data_aug/" + str(pic) + "_" + _file

                # print("    >>>>>>>>>>>  new_path:", new_path)

                # CV2 loads as BGR instead of RGB so fix that
                # try to fix BGR/RGB problem in preprocessing to save more time
                temp_img = cv2.imread(temp_path)

            # print("random temp_img.shape", temp_img.shape)
            # b, g, r = cv2.split(temp_img)         # get b,g,r
            # temp_img = cv2.merge([r, g, b])    # switch it to rgb

            label = int(x_data[0].split("/")[-1].split(".")[0].split("_")[-1]) - 1

            # print("new path:", new_path, "label:", label)
            # labels aren't zero based so don't forget to fix that
            y[idx] = label

            ######################
            ### Imagenet Stuff ###
            ######################
            temp_img = temp_img / 255.0   # convert to float

            # remove imagenet means from values
            temp_img[..., 0] -= (103.939 / 255.0)    # red
            temp_img[..., 1] -= (116.779 / 255.0)    # green
            temp_img[..., 2] -= (123.68 / 255.0)     # blue

        # print("shape:", temp_img.shape, "sample values:", temp_img[0:3, 0:3, 0])
        # print("x_data[0]", x_data[0], "y label", y[idx])

        # shape: (299, 299, 3) sample values: [[0.55318039 0.55710196 0.54533725]
        #  [0.54141569 0.55318039 0.54141569]
        #  [0.52965098 0.55710196 0.54533725]]
        # x_data[0] ../data/stage3_imgs/104794_92.jpg y label 91
        # shape: (299, 299, 3) sample values: [[-0.24681961 -0.24681961 -0.24681961]
        #  [-0.24681961 -0.24681961 -0.24681961]
        #  [-0.24681961 -0.24681961 -0.25074118]]
        # x_data[0] ../data/stage3_imgs/43233_89.jpg y label 88

        X[idx, ] = temp_img

        return [X, X_meta], keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __data_generation(self, x_data_slice):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((x_data_slice.shape[0], *self.dim, self.n_channels))
        # print("x_data_slice type:", type(x_data_slice))
        X_meta = x_data_slice[:, 1:]
        y = np.empty((x_data_slice.shape[0]), dtype=int)

        # Generate data
        for i, x_data in enumerate(x_data_slice):
            # print("x_data 0", x_data[0], "type", type(x_data[0]))
            # ID:
            # ../data/stage1_imgs/flip_116297_85.jpg

            ##################################
            ### Load from Center crop file ###
            ##################################

            temp_path = "../data/stage3_imgs/" + x_data[0].split("/")[-1]

            # For some reason CV2 loads as BGR instead of RGB
            temp_img = cv2.imread(temp_path)

            # print("temp", type(temp))
            # b, g, r = cv2.split(temp)         # get b,g,r
            # temp_img = cv2.merge([r, g, b])    # switch it to rgb

            # Store class
            # y[i] = self.labels[ID]
            y[i] = int(x_data[0].split("/")[-1].split(".")[0].split("_")[-1]) - 1

            # keep this because this kind of error could creep back in if
            # more images were downloaded/added

            # do some error checking because some files suck
            # for example some files are shape(1, 1, 3)
            # if temp.shape[0] is 1 or temp.shape[1] is 1:
            #     print("shape", temp.shape)
            #     pass    # skip this one
            # else:

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

            if temp_img.shape[0] != 299 or temp_img.shape[1] != 299:
                print("Shape isn't correct, resize", temp_img.shape)
                # else:
                #     # just get the middle of the image, resized to 299^2
                #     temp_img = crop_image(temp_img)
                #     # print("ID:", ID, "Label:", y[i])
                #
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
            else:   # make the result called 'resize' in either case
                resized = temp_img

            ######################
            ### Imagenet Stuff ###
            ######################
            resized = resized / 255.0   # convert to float

            # remove imagenet means from values
            resized[..., 0] -= (103.939 / 255.0)    # red
            resized[..., 1] -= (116.779 / 255.0)    # green
            resized[..., 2] -= (123.68 / 255.0)     # blue

            # print("sample values:", resized[0:3, 0:3, 0])
            # print("x_data[0]", x_data[0], "y label", y[i])

            # sample values: [[0.47082745 0.47082745 0.47082745]
            #  [0.47082745 0.47082745 0.47082745]
            #  [0.47082745 0.47082745 0.47082745]]
            # x_data[0] ../data/stage3_imgs/134984_82.jpg y label 81
            # sample values: [[-0.04681961 -0.04681961 -0.04681961]
            #  [-0.04681961 -0.04681961 -0.04681961]
            #  [-0.04681961 -0.04681961 -0.04681961]]
            # x_data[0] ../data/stage3_imgs/183592_76.jpg y label 75

            X[i, ] = resized

        return [X, X_meta], keras.utils.to_categorical(y, num_classes=self.n_classes)


class FireBot(keras.utils.Sequence):
    'Generates data for Model Predictions, for evaluation'
    def __init__(self, x_data, batch_size=32, dim=(299, 299), n_channels=3,
                 n_classes=128, augment=False, shuffle=False, Raven=False):
        '''
        Initialization
        Already shuffled by get_skfold_indicies so no need to shuffle again
        '''
        self.dim = dim
        self.batch_size = batch_size
        # self.list_IDs = list_IDs
        self.x_data = x_data
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.len = None
        self.augment = augment
        self.Raven = Raven
        self.on_epoch_end()
        # self.dim = dim
        # self.batch_size = batch_size
        # self.list_IDs = list_IDs
        # self.n_channels = n_channels
        # self.n_classes = n_classes
        # self.shuffle = shuffle
        # self.on_epoch_end()
        # self.len = None
        # self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        'Denotes the number of batches per epoch'
        # add one to the len to account for leftovers, then on getitem check
        # to see if it's the last batch
        self.len = int(np.floor(len(self.x_data) / self.batch_size)) + 1
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
        x_data_slice = np.array([self.x_data[k] for k in indexes])

        # print("len list_IDs_temp", len(list_IDs_temp))

        X = self.__data_generation(x_data_slice)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x_data))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, x_data_slice):
        'Generates data containing batch_size samples'

        # Initialization
        X = np.empty((x_data_slice.shape[0], *self.dim, self.n_channels))
        # print("x_data_slice type:", type(x_data_slice))

        if self.Raven is True:
            X_meta = x_data_slice[:, 1:]

        # interesting bug in initialization where X np.empty was initialized
        # with batch size instead of the x_data_slice size
        # print("pic shape", X.shape, "meta shape", X_meta.shape)

        # Generate data
        for i, x_data in enumerate(x_data_slice):
            # print("x_data 0", x_data[0], "type", type(x_data[0]))
            # ID:
            # ../data/stage1_imgs/flip_116297_85.jpg

            ###########################
            ### Load raw test image ###
            ###########################

            if self.Raven is True:
                # For some reason CV2 loads as BGR instead of RGB
                temp_img = cv2.imread(x_data[0])
            else:
                temp_img = cv2.imread(x_data)

            b, g, r = cv2.split(temp_img)      # get b,g,r
            temp_img = cv2.merge([r, g, b])    # switch it to rgb

            # basic center crop
            temp_img = crop_image(temp_img)

            # recreate stage3_imgs training data
            temp_img = cv2.resize(temp_img, self.dim, interpolation=cv2.INTER_AREA)

            # for normal training with augmentation
            if self.augment is True:

                # roll the dice on augmentation, 2 because not inclusive like lists
                _flip = np.random.randint(0, 2)

                # pre_flip = temp_img.copy()

                if _flip is 1:
                    temp_img = temp_img[:, ::-1, :]  # flip image over vertical axis

                # rotate?
                _rot = np.random.randint(0, 2)
                # trying to measure how changes to augmentation... change things
                # _rot = 0

                # pre_rot = temp_img.copy()

                if _rot is 1:
                    # figure out how much rotation (should be about -15 to 15 degrees
                    # but a normal distribution centered on 0)
                    _rotation = np.random.normal(0, 1.0)

                    # rotate and THEN also crop so there's not a lot of black on the
                    # edges
                    temp_img = rotate_bound(temp_img, _rotation)

                # pre_crop = temp_img.copy()

                # get the sub-crop of the bigger image
                temp_img = smart_crop(temp_img)

                # do fancy PCA color augmentation
                temp_img = fancy_pca(temp_img, alpha_std=0.10)

            # resize to 299 for proper prediction
            try:
                # image is just a square but not necessarily 299px x 299px
                # generalize to self.dim so other 'nets can be used, like VGG16
                resized = cv2.resize(temp_img, self.dim, interpolation=cv2.INTER_AREA)
            except:
                print("something went horribly wrong in LoaderBot _generate_data", x_data, "shape", temp_img.shape)
                # print("pre_flip", pre_flip.shape)
                # print("pre_rot", pre_rot.shape)
                # print("pre_crop", pre_crop.shape)
                print("new size:", temp_img.shape)

            ######################
            ### Imagenet Stuff ###
            ######################
            resized = resized / 255.0   # convert to float

            # remove imagenet means from values
            resized[..., 0] -= (103.939 / 255.0)    # red
            resized[..., 1] -= (116.779 / 255.0)    # green
            resized[..., 2] -= (123.68 / 255.0)     # blue

            X[i, ] = resized

        if self.Raven is True:
            return [X, X_meta]
        else:
            return X
