

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

import sys, os, multiprocessing
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import cv2
import glob


def process_image(path, out_path, flip=False, size=(299, 299)):
    '''
    Inputs:
    path: is the file path to the source image file.
    size: a tuple that specifies the output image (299, 299) for inceptionv3

    Output:
    Writes an image of the specified size to out_path
    '''
    img = cv2.imread(path)

    resized = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)

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


def clean_file_at_location(inpath, outpath="../data/stage1_imgs/"):
    '''
    input:
    in_path: directory to look for files to fix
    out_path: where do we save the fixed files
    '''

    augment_labels = [106, 19, 105, 115, 110, 58, 109, 114, 47, 127, 34, 35,
                      57, 62, 86, 74, 41, 85, 77, 25, 9, 121, 124, 66, 83]

    # in_path
    # "../data/stage1_imgs/91252_83.jpg"
    file_name = inpath.split("/")[-1]
    outpath = outpath + file_name

    # see if the file type is in an augmentation class
    if int(file_name.split("_")[-1].split(".")[0]) in augment_labels:
        # this is an augmentation class file so flip it
        process_image(inpath, outpath, flip=True)
    else:
        # this is not an augmentation class file
        process_image(inpath, outpath)


def Run():
    if len(sys.argv) != 3:
        print('Syntax: %s <train|validation|test.json> <output_dir/>' % sys.argv[0])
        sys.exit(0)

    in_dir, out_dir = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # all file paths, as a list
    file_paths = glob.glob(in_dir + "*.jpg")

    _len = len(file_paths)

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
