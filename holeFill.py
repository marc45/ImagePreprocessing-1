#! /usr/bin/env python

import cv2
import numpy as np


def hole_fill(im):
    height, width = im.shape
    im_floodfill = im.copy()

    mask = np.zeros((height+2, width+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    im_out = im | im_floodfill_inv
    return im_out


if __name__ == "__main__":
    im = cv2.imread("./img/tags/4/mask/009_mask.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    ret, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("raw", im)
    cv2.waitKey(0)
    im_out = holl_fill(im)
    cv2.imshow("hole filled", im_out)
    cv2.waitKey(0)
