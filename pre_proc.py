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


# image enhance, add brightness, curve adjust
def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    # brightness = 1.3 (1-mean/255) * 3
    brightness = 158 / mean - 0.1286
    brightness = 2 if brightness > 2 else brightness
    print(brightness)

    im = Image.fromarray(image)

    # enhance brightness
    im = ImageEnhance.Brightness(im).enhance(brightness)

    # enhance sharpness
    enh_sha = ImageEnhance.Sharpness(im)
    sharpness = 3.0
    im = enh_sha.enhance(sharpness)

    # enhance contrast
    enh_con = ImageEnhance.Contrast(im)
    contrast = 2
    im = enh_con.enhance(contrast)

    # save image
    # res_save = np.array(res)
    # cv2.imwrite("temp/"+im_file.split("/")[-1], res_save)

    # curve adjust
    im = np.array(im)
    im = curve_adjust(os.path.join(os.path.dirname(__file__), 'curve.acv'), im)
    cv2.imshow("enhance", im)
    cv2.waitKey(0)

    # to gray
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # hue, saturation, value = cv2.split(hsv)

    # B, G, R = cv2.split(image)
    # gray = 0.55*G  + 0.45*R
    # gray = gray.astype(int)

    return np.array(gray)


# big block eleminate
def big_block_eliminate(binary):
    # big erosion
    erosion = cv2.erode(binary, np.ones((19, 19), np.uint8), iterations=1)

    # minus
    erosion = binary - erosion

    return erosion


# pre process
def pre_proc(image, debug=False):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # enhancce image
    gray = enhance_image(image)
    if debug:
        cv2.imshow("gray", gray)
        cv2.waitKey(0)

    # to gray and reverse
    gray = cv2.bitwise_not(gray)

    # blur
    blur = cv2.GaussianBlur(gray, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    # to binary image
    # binary = cv2.threshold(blur, BINARY_THRESH, 255, 0)[1]
    ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if debug:
        cv2.imshow("thresh", binary)
        cv2.waitKey(0)

    # dilation
    dilation = cv2.dilate(binary, np.ones((4, 2), np.uint8), iterations=1)
    if debug:
        cv2.imshow("dilation", dilation)
        cv2.waitKey(0)

    # erosion
    erosion = cv2.erode(dilation, np.ones((3, 3), np.uint8), iterations=1)
    if debug:
        cv2.imshow("erosion", erosion)
        cv2.waitKey(0)

    return erosion


# use sobel to find the contour
def pre_proc_sobel(im_file, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

    dilation = cv2.dilate(binary, element2, iterations=1)
    cv2.imshow("dilation", dilation)
    cv2.waitKey(0)
    erosion = cv2.erode(dilation, element1, iterations=1)
    cv2.imshow("erosion", erosion)
    cv2.waitKey(0)

    dilation2 = cv2.dilate(erosion, element2, iterations=3)
    cv2.imshow("dilation2", dilation2)
    cv2.waitKey(0)

    return erosion


# skew detect
def adjust_skew(image):
    region = []
    angle = 0
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        if(area < 100):
            continue
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        cv2.approxPolyDP(cnt, epsilon, True)

        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)

        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        if height < width * 1.0 or height > width * 2.0:
            continue
        if height < 30:
            continue

        angle += rect[-1]
        region.append(box)
    print("skew angle: {}".format(angle))
    return region


def proc(image, mask, debug=False):
    affined_img = affine_transform(image, mask, debug)
    if debug:
        cv2.imshow("affined", affined_img)
        cv2.waitKey(0)
    binary = pre_proc(affined_img, debug)
    height, width = binary.shape
    binary = cv2.resize(binary, (100*width/height, 100), interpolation=cv2.INTER_CUBIC)
    if debug:
        cv2.imshow("result", binary)
        cv2.waitKey(0)


    return binary


# process
def test_proc(im_file, image, mask):
    # resize
    height, width, _ = image.shape
    image = cv2.resize(image, (100*width/height, 100), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (100*width/height, 100), interpolation=cv2.INTER_CUBIC)

    cv2.imshow("raw", image)
    cv2.waitKey(0)

    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    binary = proc(image, mask, debug=True)
    cv2.imwrite('temp/'+im_file.split('/')[-1], binary)
    return binary
    # region = adjust_skew(binary_im)
    # for box in region:
    #     cv2.drawContours(image, [box], 0, (0,255,0), 2)
    # cv2.imshow("img", image)
    # cv2.waitKey(0)


if __name__ == "__main__":
    tags_folder = "./img/tags/3/tag"
    mask_folder = "./img/tags/3/mask"

    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
        mask = cv2.imread(sys.argv[2], cv2.CV_LOAD_IMAGE_GRAYSCALE)
        test_proc(sys.argv[1].split('/')[-1], image, mask)
    else:
        tag_files = os.listdir(tags_folder)
        for tag_file in tag_files:
            image = cv2.imread(os.path.join(tags_folder, tag_file))
            mask = cv2.imread(os.path.join(mask_folder, tag_file.split('.')[0]+"_mask.jpg"), cv2.CV_LOAD_IMAGE_GRAYSCALE)
            test_proc(tag_file, image, mask)

#
# (h, w) = image.shape[:2]
# center = (w//2, h//2)
# M = cv2.getRotationMatrix2D(center, angle, 1.0)
#
# rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#
# cv2.putText(rotated, "angle: {:.2f} degree".format(angle), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
#
# cv2.imshow("Input", image)
# cv2.imshow("rotate", rotated)
# cv2.waitKey(0)
#
#
#
#
# # tesser.pytesseract.tesseract_cmd = '/usr/share/tesseract-ocr/tessdata/'
# # tessdata_dir_config = "/usr/share/tesseract-ocr/tessdata"
# image = Image.open(im_file)
# code = tesser.image_to_string(image, config='-psm 7 my_digits')# , config=tessdata_dir_config)
# print code
