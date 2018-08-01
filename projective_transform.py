#! /usr/bin/env python

import os
import cv2
import numpy as np


def find_contour(img, mask):
    mask_copy = mask.copy()
    contours, hierarchy = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape

    minarea = 500
    maxarea = 100000
    width_height_ratio = [1, 20]
    tag_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minarea:
            continue
        if area > maxarea:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w/h < width_height_ratio[0] or w/h > width_height_ratio[1]:
            continue

        x = x-5 if x >= 5 else x
        y = y-5 if y >= 5 else y
        x1 = x+w+10 if x+w+10 < width else width-1
        y1 = y+h+10 if y+h+10 < height else height-1

        tag_boxes.append([x, y, x1, y1])

    return tag_boxes


def affine_transform(im, mask, debug=False):
    ret, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])

    # remember always width > height
    center = rect[0]
    roi_width, roi_height = rect[1]
    angle = rect[2]

    if cv2.__version__.split('.')[0] == "2":
        box = cv2.cv.BoxPoints(rect)
    else:
        box = cv2.boxPoints(rect)
    box = np.int0(box)
    if debug:
        cv2.line(im, tuple(box[0]), tuple(box[1]), [0,0,255], 2)
        cv2.line(im, tuple(box[1]), tuple(box[2]), [0,0,255], 2)
        cv2.line(im, tuple(box[2]), tuple(box[3]), [0,0,255], 2)
        cv2.line(im, tuple(box[3]), tuple(box[0]), [0,0,255], 2)
        cv2.imshow("im", im)
        cv2.waitKey(0)

    if angle > 45:
        angle = angle - 90
        roi_width = rect[1][1]
        roi_height = rect[1][0]

    elif angle < -45:
        angle = angle + 90
        roi_width = rect[1][1]
        roi_height = rect[1][0]

    height, width = mask.shape
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img = cv2.warpAffine(im, M, (width, height))

    roi = img[int(center[1]-roi_height/2): int(center[1]+roi_height/2), int(center[0]-roi_width/2):int(center[0]+roi_width/2)]
    if debug:
        cv2.imshow("roi", roi)
        cv2.waitKey(0)
    return roi


def proc():
    tag_files = os.listdir("./img/tags/tag")
    for tag_file in tag_files:
        tag_file = os.path.join("./img/tags/tag", tag_file)
        mask_file = os.path.join("./img/tags/mask", tag_file.split('/')[-1].split('.')[0]+"_mask.jpg")
        tag = cv2.imread(tag_file)
        mask = cv2.imread(mask_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        affine_transform(tag, mask, debug=True)


if __name__ == "__main__":
    proc()
    # raw = cv2.imread("raw.jpg")
    # mask = cv2.imread("binary.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)


    # tag_boxes = find_contour(raw, mask)

    # for box in tag_boxes:
    #     mask_roi = mask[box[1]:box[3], box[0]:box[2]]
    #     raw_roi = raw[box[1]:box[3], box[0]:box[2]]
    #     affine_transfer(raw_roi, mask_roi)
