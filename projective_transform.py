#! /usr/bin/env python

import os
import cv2
import numpy as np
from extractInnerMaxRect import findRotMaxRect
from holeFill import hole_fill
import math

OPENCV3 = True if cv2.__version__.split('.')[0] == "3" else False


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


def find_max_inner_rect(mask):
    idx_in = np.where(mask > 200)
    # idx_out = np.where(mask < 50)
    mask = np.ones_like(mask)
    mask[idx_in] = 0
    rect_coord_ori, angle, coord_out_rot = findRotMaxRect(mask, flag_opt=True, nbre_angle=5,
                                                          flag_parallel=False,
                                                          flag_out='rotation',
                                                          flag_enlarge_img=False,
                                                          limit_image_size=150)

    center_x = 0
    center_y = 0
    for coord in rect_coord_ori:
        center_x += coord[1]
        center_y += coord[0]

    width = int(math.sqrt((rect_coord_ori[0][0] - rect_coord_ori[1][0])**2 +
                          (rect_coord_ori[0][1] - rect_coord_ori[1][1])**2) + 0.5)
    height = int(math.sqrt((rect_coord_ori[1][0] - rect_coord_ori[2][0])**2 +
                           (rect_coord_ori[1][1] - rect_coord_ori[2][1])**2) + 0.5)
    center_x = center_x / 4.0
    center_y = center_y / 4.0

    print(angle)
    return ((center_x, center_y), (width, height), angle)


def find_min_enclosing_rect(mask):
    if OPENCV3:
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 1:
        return False
    elif len(contours) == 1:
        rect = cv2.minAreaRect(contours[0])
    else:
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > max_area:
                max_area = area
                final_contour = c
        rect = cv2.minAreaRect(final_contour)

    # remember in rect always width > height
    center = rect[0]
    roi_width, roi_height = rect[1]
    angle = rect[2]

    if angle > 45:
        angle = angle - 90
        roi_width = rect[1][1]
        roi_height = rect[1][0]

    elif angle < -45:
        angle = angle + 90
        roi_width = rect[1][1]
        roi_height = rect[1][0]

    return (center, (int(roi_width), int(roi_height)), angle)


def affine_transform(im, mask, debug=False):
    height, width = mask.shape
    im = cv2.resize(im, (40*width/height, 40), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (40*width/height, 40), interpolation=cv2.INTER_CUBIC)
    height, width = mask.shape

    ret, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    # fille holes
    mask = hole_fill(mask)

    # dilate and erode
    mask = cv2.erode(mask, np.ones((5, 7), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    if debug:
        cv2.imshow("mask_affined", mask)
        cv2.waitKey(0)
    # find minimum enclosing rectangle
    rect = find_min_enclosing_rect(mask)
    # find max inner rectangle
    # rect = find_max_inner_rect(mask)

    if rect == False:
        return False

    if debug:
        im_copy = im.copy()
        if OPENCV3:
            box = cv2.boxPoints(rect)
        else:
            box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.line(im_copy, tuple(box[0]), tuple(box[1]), [0, 0, 255], 1)
        cv2.line(im_copy, tuple(box[1]), tuple(box[2]), [0, 0, 255], 1)
        cv2.line(im_copy, tuple(box[2]), tuple(box[3]), [0, 0, 255], 1)
        cv2.line(im_copy, tuple(box[3]), tuple(box[0]), [0, 0, 255], 1)
        cv2.imshow("roi", im_copy)
        cv2.waitKey(0)

    center, (roi_width, roi_height), angle = rect
    M = cv2.getRotationMatrix2D(center, angle/2.0, 1)
    img = cv2.warpAffine(im, M, (width, height))

    # height_low = center[1] - roi_height/2 if center[1] - roi_height/2 > 0 else 1
    # height_high = center[1] + roi_height/2 if center[1] + roi_height/2 < 0 else 1

    # roi = img[int(center[1]-roi_height/2): int(center[1]+roi_height/2),
    #           int(center[0]-roi_width/2): int(center[0]+roi_width/2)]
    roi = img

    roi = cv2.resize(roi, (100*roi_width/roi_height, 100), interpolation=cv2.INTER_CUBIC)
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
