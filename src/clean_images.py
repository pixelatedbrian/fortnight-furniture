

#!/usr/bin/python3.5
# -*- coding:utf-8 -*-
# Created Time: Fri 02 Mar 2018 03:58:07 PM CST
# Purpose: download image
# Mail: tracyliang18@gmail.com
# Adapted to python 3 by Aloisio Dourado in Sun Mar 11 2018

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

##################
### HOW TO USE ###
##################
# modify hardcoded destination path in this script first!
# def clean_file_at_location(inpath, outpath="../data/stage2_imgs/"):
#                                            ^^^^^^^^^^^^^^^^^^^^^^
# THEN run in local directory like:
# python clean_images.py ../data/imgs/

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


def clean_file_at_location(inpath, outpath="../data/stage3_imgs/"):
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
