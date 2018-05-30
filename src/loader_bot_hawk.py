import numpy as np
import keras
import cv2
from image_aug import fancy_pca, rotate_bound, smart_crop, crop_image

# HAWK iteration
# feeds Inceptionv3 based DNN but without the Raven network concatenated


class LoaderBot(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img_list, y_labels, batch_size=32, dim=(299, 299),
                 n_channels=3, n_classes=128, shuffle=False,
                 augment_odds=0.0, augment=False):
        '''
        Initialization
        Already shuffled by get_skfold_indicies so no need to shuffle again
        '''
        self.dim = dim                      # h x w of image
        self.batch_size = batch_size        # batch size
        self.labels = y_labels              # y labels for learning
        self.img_list = img_list            # files to load
        self.n_channels = n_channels        # number of colors
        self.n_classes = n_classes          # categories to one-hot y labels to
        self.shuffle = shuffle              # shuffle indices each time?
        self.len = None                     # how many rows of data total
        self.augment = augment              # use augmentation? (for test no)
        self.augment_odds = augment_odds    # probability of loading an augmented image
        self.on_epoch_end()                 # shuffle, if desired, at epoch end

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # since there is augmentation add the same items to a bigger list
        # as many extra times as augmentation is, ie 2x or 10x
        self.indexes = np.arange(len(self.img_list))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_slice):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(file_slice), *self.dim, self.n_channels))
        y = np.empty((len(file_slice)), dtype=int)

        # Generate data
        for idx, img_file in enumerate(file_slice):
            # img_file:
            # ../data/stage1_imgs/flip_116297_85.jpg
            # temp = cv2.imread(img_file)

            # is augment on or off?
            if self.augment is True:

                # roll to see if we actually pick an augmented file
                roll = np.random.random()

                if roll < self.augment_odds:    # if odds = 1.0 then should be true

                    # select a random image from augmentation then
                    pic_num = str(np.random.randint(0, 10))

                    # build path from that random pic_num
                    path = "../data/static_aug2/" + pic_num + "_" + img_file.split("/")[-1]

                else:   # don't augment
                    path = "../data/stage3_imgs/" + img_file.split("/")[-1]

            else:   # don't augment
                path = "../data/stage3_imgs/" + img_file.split("/")[-1]

            # load whatever the file is:
            img_data = cv2.imread(path)

            if img_data.shape[0] != self.dim[0]:
                # resize images for certain networks besides InceptionV3
                try:
                    # image is just a square but not necessarily 299px x 299px
                    # generalize to self.dim so other 'nets can be used, like VGG16
                    img_data = cv2.resize(img_data, self.dim, interpolation=cv2.INTER_AREA)
                except:
                    print("something went horribly wrong in LoaderBot _generate_data", img_file, "shape", img_data.shape)

            # if img_data is None:
            #     print(">>>>>>>>>>>>>>>>", path)

            # OpenCV doesn't use RGB by default :-(
            # b, g, r = cv2.split(img_data)         # get b,g,r
            # img_data = cv2.merge([r, g, b])    # switch it to rgb

            ######################
            ### Imagenet Stuff ###
            ######################
            img_data = img_data.astype('float64')
            img_data = img_data / 255.0  # convert to float

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            img_data[..., 0] = (img_data[..., 0] - mean[0]) / std[0]  # red
            img_data[..., 1] = (img_data[..., 1] - mean[1]) / std[1]  # green
            img_data[..., 2] = (img_data[..., 2] - mean[2]) / std[2]  # blue

            # remove imagenet means from values
            # img_data[..., 0] = (img_data[..., 0] - 103.939) * 0.017    # red
            # img_data[..., 1] = (img_data[..., 1] - 116.779) * 0.017    # green
            # img_data[..., 2] = (img_data[..., 2] - 123.68) * 0.017     # blue

            X[idx, ] = img_data

            y[idx] = int(path.split("/")[-1].split(".")[0].split("_")[-1]) - 1

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        self.len = int(np.floor(len(self.img_list) / self.batch_size))
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
        img_list_slice = [self.img_list[k] for k in indexes]

        # print("len list_IDs_temp", len(list_IDs_temp))

        # Generate data
        X, y = self.__data_generation(img_list_slice)

        return X, y


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
