#! /usr/bin/env python

import sys
# import pytesseract as tesser
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import cv2
import os
from curve_proc import curve_adjust
# from contrast import preprocess
from projective_transform import affine_transform

GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
BINARY_THRESH = 160

folder = "./entire_image/2"

for im_name in os.listdir(folder):
    im_file = os.path.join(folder, im_name)
    im = cv2.imread(im_file)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    brightness = (1-mean/255) * 3

    image = Image.fromarray(im)
    # enhance brightness
    image = ImageEnhance.Brightness(image).enhance(brightness)

    # save image
    image_save = np.array(image)
    cv2.imwrite(os.path.join(folder, im_name.split('.')[0]+"_out.jpg"), image_save)


    # enhance contrast
    # enh_con = ImageEnhance.Contrast(im)
    # contrast = 5
    # im = enh_con.enhance(contrast)

    # curve adjust
    # im = np.array(im)
    # im = curve_adjust(os.path.join(os.path.dirname(__file__), 'curve.acv'), im)

    # to gray
    # hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # hue, saturation, value = cv2.split(hsv)
