#! /usr/bin/env python


import cv2
import numpy as np


GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
ADAPTIVE_THRESH_BLOCK_SIZE = 27
ADAPTIVE_THRESH_WEIGHT = 13


# extract value of image
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    imgHue, imgSaturation, imgVaule = cv2.split(imgHSV)

    return imgVaule


def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat

def dilateAndErode(binary):
    erosion = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    dilation = cv2.dilate(erosion, np.ones((5, 5), np.uint8), iterations=1)
    # erosion = cv2.erode(dilation, np.ones((4, 4), np.uint8), iterations=1)
    cv2.imshow("erosion", erosion)
    cv2.waitKey(0)
    # dilation = cv2.dilate(erosion, np.ones((5, 5), np.uint8), iterations=1)
    cv2.imshow("dilation", dilation)
    cv2.waitKey(0)

# preprocess
def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    height, width = imgGrayscale.shape
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,
                                      ADAPTIVE_THRESH_BLOCK_SIZE,
                                      ADAPTIVE_THRESH_WEIGHT)
    cv2.imshow("gray", imgGrayscale)
    cv2.waitKey(0)
    cv2.imshow("maxcontrastgray", imgMaxContrastGrayscale)
    cv2.waitKey(0)
    cv2.imshow("binary", imgThresh)
    cv2.waitKey(0)

    dilateAndErode(imgThresh)
    return imgThresh
