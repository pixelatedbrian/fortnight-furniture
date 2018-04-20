

#!/usr/bin/python3.5

# Process RAW downloaded images into 299x299 images ready for InceptionV3
# Version 2.0+ adds extended image augmentation. The reason that augmentation
# is being performed in a permanent manner here, as opposed to dynamic augmentation
# on demand, is because images that have aspect ratios that are non-square have
# additional information potential that is not available to a cropped square image.

# Ie: If an image is 2x as wide as it is tall version 1.x will take the middle
# square of that image and ignore the left 0.5 section of the image as well as
# the right 0.5 part of the image.

# Additionally, most of the source images are well above the 299x299 resolution
# and so are downscaled. In my opinion this provides an opportunity to crop at
# native resolution and then downsample to 299px square, effectively causing a
# zoom but also maintaining higher detail and resolution for the model to learn
# from. This being relative to an upsampled and slightly zoomed image from
# typical augmentation.

# Finally, dynamic image augmentation can then be applied to these native sourced
# images to further extend the data augmentation. Although computationally, for
# my resources, it's not clear how tenable that would be.

##################
### HOW TO USE ###
##################
# modify hardcoded destination path in this script first!
# def clean_file_at_location(inpath, outpath="../data/stage2_imgs/"):
#                                            ^^^^^^^^^^^^^^^^^^^^^^
# THEN run in local directory like:
# python clean_images.py ../data/imgs/ ../data/stage2_imgs/

# 2.0
# add 'aug_' leader to all augmented images
# new augmentation pipeline: ten total images per original image, counting the original
# 1: the image cropped and scaled to 299px square
# 2: 75-95% of the square, randomly positioned
# 3: same as 2 but new random selection
# 4: rotate randomly -15 to 15 degrees, then random crop like 2
# 5: repeat
# 6: flip 1
# 7-8: repeat #2 subselect alg on flipped image
# 9-10: repeat #4 rotated subselect alg on flipped image

# v 1.2
# perform simple flip augmentation to all images

# v1.1
# crop images instead of weird scaling only

# v1.0
# rescale all images to 1:1 ratio
# resize images to 299x299

import sys, os, multiprocessing
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import cv2
import glob


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

        return crop_image

    elif img.shape[1] < img.shape[0]:
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

    else:   # should never fire
        print("image_crop(img): huh, whoops")
        return img


def process_image(path, out_path, flip=False, size=(299, 299)):
    '''
    Inputs:
    path: is the file path to the source image file.
    size: a tuple that specifies the output image (299, 299) for inceptionv3

    Output:
    Writes an image of the specified size to out_path
    '''
    img = cv2.imread(path)

    debug = img.shape

    # figure out the aspect ratio of the image, if it's pretty square
    # then just resize as is, otherwise send to crop_image to take out the
    # middle square of the image
    ratio = img.shape[0] / img.shape[1]

    if ratio < 0.98 or ratio > 1.02:
        img = crop_image(img)

    try:
        resized = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
    except:
        print("process_image: something went wrong when trying to resize: ", debug)
        return

    pil_image = Image.fromarray(resized)

    try:
        pil_image.save(out_path, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image {:}'.format(out_path))
        return

    if flip is True:
        # reverse image with numpy slicing
        resized = resized[:, ::-1, :]

        pil_image = Image.fromarray(resized)

        # figure out correct filename
        # ex: "../data/stage1_imgs/test.jpg"
        temp = out_path.split("/")

        temp[-1] = "flip_" + temp[-1]   # last item should be test.jpg
        new_out_path = "/".join(temp)

        try:
            pil_image.save(new_out_path, format='JPEG', quality=90)
        except:
            print('Warning: Failed to save image {:}'.format(new_out_path))
            return


def clean_file_at_location(inpath, outpath="../data/stage4_imgs/"):
    '''
    input:
    in_path: directory to look for files to fix
    out_path: where do we save the fixed files
    '''

    # comment out in v1.1, remove in v1.2 if it helps
    # augment_labels = [106, 19, 105, 115, 110, 58, 109, 114, 47, 127, 34, 35,
    #                   57, 62, 86, 74, 41, 85, 77, 25, 9, 121, 124, 66, 83]

    # in_path
    # "../data/stage1_imgs/91252_83.jpg"
    file_name = inpath.split("/")[-1]
    outpath = outpath + file_name

    # flip no images as of v1.1, flip all images in 1.2
    process_image(inpath, outpath, flip=True)

    # # see if the file type is in an augmentation class
    # if int(file_name.split("_")[-1].split(".")[0]) in augment_labels:
    #     # this is an augmentation class file so flip it
    #     process_image(inpath, outpath, flip=True)
    # else:
    #     # this is not an augmentation class file
    #     process_image(inpath, outpath)


def Run():
    if len(sys.argv) != 3:
        print('Syntax: %s <train|validation|test.json> <output_dir/>' % sys.argv[0])
        sys.exit(0)

    in_dir, out_dir = sys.argv[1:]

    print("in_dir", in_dir)
    print("out_dir", out_dir)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # all file paths, as a list
    file_paths = glob.glob(in_dir + "*.jpg")

    _len = len(file_paths)

    print("len of file_paths", _len)

    # for _file in file_paths[:100]:
    #     clean_file_at_location(_file)

    pool = multiprocessing.Pool(processes=40)

    with tqdm(total=_len) as t:
        for _ in pool.imap_unordered(clean_file_at_location, file_paths):
            t.update(1)

    #
    # key_url_list = ParseData(data_file)
    # pool = multiprocessing.Pool(processes=100)
    #
    # with tqdm(total=len(key_url_list)) as t:
    #     for _ in pool.imap_unordered(DownloadImage, key_url_list):
    #         t.update(1)


if __name__ == '__main__':
  Run()
