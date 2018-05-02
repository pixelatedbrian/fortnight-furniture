import cv2     # opencv3
import glob    # to get lists of matching files
import numpy as np

from PIL import Image
from io import BytesIO


def fancy_pca(img, alpha_std=100):
    '''
    take in an image and does PCA to augment the colors while maintaing the structure of
    the image.

    This is 'Fancy PCA' from:
    # http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    #######################
    #### FROM THE PAPER ###
    #######################
    "The second form of data augmentation consists of altering the intensities
    of the RGB channels in training images. Specifically, we perform PCA on the
    set of RGB pixel values throughout the ImageNet training set. To each
    training image, we add multiples of the found principal components, with
    magnitudes proportional to the corresponding eigenvalues times a random
    variable drawn from a Gaussian with mean zero and standard deviation 0.1.
    Therefore to each RGB image pixel Ixy = [I_R_xy, I_G_xy, I_B_xy].T
    we add the following quantity:
    [p1, p2, p3][α1λ1, α2λ2, α3λ3].T

    Where pi and λi are ith eigenvector and eigenvalue of the 3 × 3 covariance
    matrix of RGB pixel values, respectively, and αi is the aforementioned
    random variable. Each αi is drawn only once for all the pixels of a
    particular training image until that image is used for training again, at
    which point it is re-drawn. This scheme approximately captures an important
    property of natural images, namely, that object identity is invariant to
    change."
    ### END ###############

    Other useful resources for getting this working:
    # https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4
    # https://gist.github.com/akemisetti/ecf156af292cd2a0e4eb330757f415d2

    Inputs:
    img:  numpy array with (h, w, rgb) shape, as ints between 0-255)
    alpha_std:  how much to perturb/scale the eigen vecs and vals
                the paper used std=0.1 but having cranked it up to 100 I see some color
                and hue anomolies which honestly I've seen during EDA as well.
                However this is effectively a hyperparameter that needs to be tuned some.

    Returns:
    numpy image-like array as float range(0, 1)
    '''

    orig_img = img.astype(float).copy()

    img = img / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

#     eig_vals [0.00154689 0.00448816 0.18438678]

#     eig_vecs [[ 0.35799106 -0.74045435 -0.56883192]
#      [-0.81323938  0.05207541 -0.57959456]
#      [ 0.45878547  0.67008619 -0.58352411]]

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    alpha = np.random.normal(0, alpha_std)

    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):   # RGB
        orig_img[..., idx] += add_vect[idx]

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)

    # about 100x faster after vectorizing the numpy, it will be even faster later
    # since currently it's working on full size images and not small, square
    # images that will be fed in later as part of the post processing before being
    # sent into the model
#     print("elapsed time: {:2.2f}".format(time.time() - start_time), "\n")

    return orig_img


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


def smart_crop(img, extra_displacement=0.0):
    '''
    Should be able to tolerate images that are tall but narrow
    as well as wide and thin.

    INPUTS:
    img: a numpy array that is an image like shape=(h, w, 3)
    extra_displacement: a multiplier that let the crop hit more remote areas
                        of the Image

    RETURNS:
    img-like numpy array that is a subset of the input array
    '''
    h, w = img.shape[0], img.shape[1]

#     print(h, w)

    if h < w:
        ratio = w / (1.0 * h)

#         size = 0.75 - 1.0
        size = np.random.normal(0.80, .07)

        # size > 1.0 causes complexity so for now just constrain to 1.0
        if size > 1.0:
            size = 0.998

        # since height is smaller use that as the constraining factor
        # pad off the grid parts with black
        size = int(h * size)

        # determine displacement range, as in how far up or down we can go
        # and be in valid space
        displacement_range = h - size

        if displacement_range < 0:
            print("\n\ndisplacement_range", displacement_range, "width:", w, "size", size, "\n\n")
            displacement_range = 1

        # figure out top left coordinates
        tl_y = (displacement_range) / 2

        # now randomly select an offset within that range:
        tl_y += np.random.randint(0, displacement_range) - (displacement_range / 2)

        # find initial tl_x
        tl_x = (w - size) / 2

        # scale displacement range to the aspect ratio
        displacement_range *= ratio

        # increase displacement in order to get further image coverage
        displacement_range += (displacement_range * extra_displacement)

        # randomly select an x offset that scales with the ratio:
        tl_x += np.random.randint(0, displacement_range) - (displacement_range / 2.0)

        # now bottom right coords should simply be top left + size
        br_x = tl_x + size

        br_y = tl_y + size

        tl_x, tl_y, br_x, br_y = int(tl_x), int(tl_y), int(br_x), int(br_y)

        if tl_y < 0:
            print(tl_x, tl_y)

        if br_y > h:
            print("h", h, "x", br_x, "y", br_y)

    else:   # h > w then
        ratio = h / (w * 1.0)

#         size = 0.75 - 1.0
        size = np.random.normal(0.80, .07)

        # size > 1.0 causes complexity so for now just constrain to 1.0
        if size > 1.0:
            size = 0.998

        # since width is smaller use that as the constraining factor
        # pad off the grid parts with black
        size = int(w * size)

        # determine displacement range, as in how far left or right we can go
        # and be in valid space
        displacement_range = w - size

        if displacement_range < 0:
            print("\n\ndisplacement_range", displacement_range, "width:", w, "size", size, "\n\n")
            displacement_range = 1

        # figure out top left coordinates
        tl_x = (displacement_range) / 2

        # now randomly select an offset in that range
        tl_x += np.random.randint(0, displacement_range) - (displacement_range / 2.0)

        # find initial tl_y
        tl_y = (h - size) / 2

        # scale displacement range to the aspect ratio
        displacement_range *= ratio

        # increase displacement in order to get further image coverage
        displacement_range += (displacement_range * extra_displacement)

        # randomly select an x offset that scales with the ratio:
        tl_y += np.random.randint(0, displacement_range) - (displacement_range / 2)

        # now bottom right coords should simply be top left + size
        br_x = tl_x + size

        br_y = tl_y + size

        tl_x, tl_y, br_x, br_y = int(tl_x), int(tl_y), int(br_x), int(br_y)

        if tl_x < 0:
            print(tl_x, tl_y)

        if br_x > w:
            print("w", w, "x", br_x, "y", br_y)

    # now return the selected slices
    return img[tl_y:br_y, tl_x:br_x, ...]
