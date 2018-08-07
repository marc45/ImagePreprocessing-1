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

im_file = "./entire_image/3/222.jpg"
save_image= "./entire_image/3_out/222.jpg"
mask_file = ["./entire_image/3/222_11.jpg", "./entire_image/3/222_12.jpg", "./entire_image/3/222_13.jpg"]


print("reading {}".format(im_file))
im = cv2.imread(im_file)

brightness = []
for im_f in mask_file:
    image = cv2.imread(im_f)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    mean = np.mean(v)

    brightness.append(mean)

bright = np.mean(np.array(brightness))
print("brightness: {}".format(bright))
bright = (1-bright/255) * 3

image = Image.fromarray(im)
# enhance brightness
image = ImageEnhance.Brightness(image).enhance(bright)

# save image
image_save = np.array(image)
cv2.imwrite(save_image, image_save)


    # enhance contrast
    # enh_con = ImageEnhance.Contrast(im)
    # contrast = 5
    # im = enh_con.enhance(contrast)

    # curve adjust
    # im = np.array(im)
    # im = curve_adjust(os.path.join(os.path.dirname(__file__), 'curve.acv'), im)
