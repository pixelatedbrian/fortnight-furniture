import numpy as np
import keras
import cv2


def crop_image(img):
    '''
    Inputs:
    img: a numpy array that is an image that has a relatively non-square StratifiedKFold

    returns:
    crop_image, square crop from the middle of the image
    '''

    if img.shape[0] < img.shape[1]:
        # height is smaller than width
        temp_size = int(img.shape[0] / 2)

        # find the midpoint of the long end
        mid_point = int(img.shape[1] / 2)

        start = mid_point - temp_size
        end = mid_point + temp_size

        # actually do the slicing
        crop_image = img[:, start:end, :]
        # print("normal ratio", crop_image.shape)

    elif img.shape[1] <= img.shape[0]:
        # width is smaller than height (weird)
        temp_size = int(img.shape[1] / 2)

        # find the midpoint of the long end
        mid_point = int(img.shape[0] / 2)

        start = mid_point - temp_size
        end = mid_point + temp_size

        # actually do the slicing
        crop_image = img[start:end, ...]
        # print("weird aspect ratio", crop_image.shape)

    return crop_image


def image_sub_select(img, min_size=None, max_size=0.99):
    '''
    Inputs:
    img: cv2/numpy array representing a loaded image
    min_size: the smallest proportion of the image to be cropped, with respect to the longer axis
                if min_size=None then figure out how big the image is relative to 299px square and
                allow zoom up to 0.5.
    max_size: the largest proportion of the image to be cropped, with respect the the longer axis

    This function, currently not part of a class, will randomly roll a zoom ratio from a uniform
    distribution between the min_size and max_size.

    Returns:
    img but in the cropped form
    '''
    h, w = img.shape[0], img.shape[1]

    constraint = 0

    # experiment with large
    # what is smaller, width or height?
    if h > w:
        constraint = w
    else:
        constraint = h

    # if the image is quite small then upsize it a bit, kind of a hack but
    # this is already a bad image
    if constraint < 350:
        scale = 350 / constraint
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        constraint = 350

#     print(img.shape)
    # (412, 550, 3)
    ogc = constraint  # remember the original size of the constraint

    # if min_size is None then figure out smallest min_size down to 0.5:
    if min_size is None:
        # how does constraint (smallest side) compare to 299px target size?
        con_ratio = 299.0 / constraint

        if con_ratio < 0.6:
            con_ratio = 0.6

        if con_ratio < 0.7:
            max_size -= 0.2

        crop_size = np.random.uniform(con_ratio, max_size)
    else:
        crop_size = np.random.uniform(min_size, max_size)

    constraint *= crop_size
#     print("crop_constraint", int(constraint))

    mid_x, mid_y = w / 2, h / 2
    tl_x, tl_y = int(mid_x - (constraint / 2)), int(mid_y - (constraint / 2))
    br_x, br_y = int(tl_x + constraint), int(tl_y + constraint)

#     print("initial top left", tl_x, tl_y)
#     print("initial bottom right", br_x, br_y)

    _range = ogc - constraint
#     print("range", int(_range))

    # figure out position within range
    x_pos = int(np.random.uniform(0, _range) - (_range / 2))
    y_pos = int(np.random.uniform(0, _range) - (_range / 2))
#     print("fuzz", x_pos, y_pos)

    tl_x, tl_y = int(tl_x + x_pos), int(tl_y + y_pos)

    # fix out of bounds for top left from using larger axis:
    if tl_x < 0:
        tl_x = 0

    if tl_y < 0:
        tl_y = 0

    br_x, br_y = int(tl_x + constraint), int(tl_y + constraint)
#     print("final top left", tl_x, tl_y)
#     print("final bottom right", br_x, br_y)
#     print("shape", img.shape)

    # fix out of bounds for bottom right from using larger axis:
    if br_x >= img.shape[1]:
        # went too far right
        # print("went too far right")

        # figure out how much too far
        diff = img.shape[1] - br_x

        # and fix
        br_x = img.shape[1] - 1
        tl_x += diff   # this is a negative value so add it
        # print("crop: shape", img.shape)
        # print("constraint", constraint)
        # print("crop: diff", diff)
        # print("crop: br_x", br_x, "br_y", br_y)
        # print("crop: tl_x", tl_x, "tl_y", tl_y)


    if br_y >= img.shape[0]:
        # went too far down
        # print("went too far down")

        # figure out how much too far
        diff = img.shape[0] - br_y

        # print("diff", diff)

        # then fix
        br_y = img.shape[0] - 1
        tl_y += diff   # this is a negative value so add it

    out = img[tl_y:br_y, tl_x:br_x, :]
    # print("shape", img.shape, "out shape", out.shape)

    return out


def random_rotation(img, min_angle=-15, max_angle=15):
    '''
    Inputs:
    img: numpy array representing an image in shape (height, width, 3)
    min_angle:  lowest possible angle to roll, typically -15
    max_angle:  highest possible angle to roll, typically 15

    returns:
    img as rotated numpy array
    '''
    angle = np.random.uniform(min_angle, max_angle)

    return rotate_bound(img, angle)   # returns a rotated, cropped img np array


def rotate_bound(image, angle):
    # resource found here:
    # https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # figure out difference in new x and y to pass back
    w_diff, h_diff = nW - w, nH - h
#     print("angle", angle, "diff", w_diff, h_diff)

    # divide the difference in half in order to figure out top left
    w_hdiff, h_hdiff = w_diff // 2, h_diff // 2

    # crop out image back to previous size to prevent massive overcrop
    temp_img = cv2.warpAffine(image, M, (nW, nH))

    out_tlx, out_tly = w_hdiff, h_hdiff
    out_brx, out_bry = w_hdiff + w, h_hdiff + h

    # perform the actual rotation and return the image
    return temp_img[out_tly:out_bry, out_tlx:out_brx, :]


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

                pre_rot = temp_img.copy()

                if _rot is 1:
                    # figure out how much rotation (should be about -15 to 15 degrees
                    # but a normal distribution centered on 0)
                    _rotation = np.random.normal() * 5 - 2.5

                    # rotate and THEN also crop so there's not a lot of black on the
                    # edges
                    temp_img = rotate_bound(temp_img, _rotation)

                pre_crop = temp_img.copy()

                # get the sub-crop of the bigger image
                temp_img = image_sub_select(temp_img)

            else:
                # just get the middle of the image, resized to 299^2
                temp_img = crop_image(temp_img)

            try:
                # image is just a square but not necessarily 299px x 299px
                resized = cv2.resize(temp_img, (299, 299), interpolation=cv2.INTER_AREA)
            except:
                print("something went horribly wrong in LoaderBot _generate_data", ID, "shape", temp.shape)
                print("pre_flip", pre_flip.shape)
                print("pre_rot", pre_rot.shape)
                print("pre_crop", pre_crop.shape)
                print("new size:", temp_img.shape)

            X[i, ] = resized

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
