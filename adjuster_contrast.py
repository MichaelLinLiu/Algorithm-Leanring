# M: This program aims to correct contrast based on the algorithm from
# https://www.cnblogs.com/Imageshop/p/9129162.html
# https://www.researchgate.net/publication/220051147_Contrast_image_correction_method?_sg=ABng1QJg-2Vt0rGjQdtqYP3IXiWGmvo8GxVa8YkYlr20TZK0lNWA2X1R2IOmFA00jnGWq5NirSUfFoq7TtAc3w
# https://pythonmana.com/2020/11/202011132156109274.html
import math
import time
import numpy as np
import cv2


# M: Step1
# M: write a global gamma correction function.
# formula: (i,j) = 255 * [i(i,j)/255] ** lambda or O = I ^ (1 / G)
# invert the gamma: 1/G
# scale the pixel intensities from [0, 255] to [0, 1.0]
# power it with the exponent which is the inverted gamma value
# scale back to 8 bit by * 255
# apply gamma correction to the whole image by using the lookup table
def mGlobalGammaAdjuster(image, gamma=1):
    invGamma = 1.0 / gamma
    table = np.array([255 * ((i / 255.0) ** invGamma)
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


# M: Step2
# M: write a local gamma correction function (Moroney suggestion: alpha is 2).
# the whole formula: (i,j) = 255 * [i(i,j)/255] ** lambda
# In Moroney algorithm, lambda is 2 ** [128 - mask(i,j)/128]
# mask(i,j) is an inverted Gaussian low-pass filter of the intensity of the input image.
def mLocalGammaAdjuster_GaussianMask(img, sd=2):
    mask = cv2.GaussianBlur(img, (5, 5), 0)[:, :, 0]
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            lam = math.pow(sd, ((128 - (255 - mask[h][w])) / 128.0))
            img[h, w] = 255 * math.pow((img[h, w][0] / 255.0), lam)
    return img


# M: Step3
# M: write a local gamma correction function by using bilateral filter.
# the whole formula: (i,j) = 255 * [i(i,j)/255] ** lambda
# Lambda is a ** [128 - BFmask(i,j)/128]
# BFmask(i,j) is an inverted bilateral filter of the intensity of the input image.
# a is smaller than 128 then a = math.log(averageI / 255) / math.log(0.5). If a is bigger than 128,a = math.log(0.5) / math.log(averageI / 255)
def mLocalGammaAdjuster_BilateralFilterMask(img):
    averageI = np.mean(img)
    if 0 < averageI <128:
        a = math.log(averageI / 255) / math.log(0.5)
    elif 128 < averageI < 255:
        a = math.log(0.5) / math.log(averageI / 255)
    else:
        a = 1
    print('a:', a)
    mask = cv2.bilateralFilter(img, 15, 75, 75)[:, :, 0]
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            lam = math.pow(a, ((128 - (255 - mask[h][w])) / 128.0))
            img[h, w] = 255 * math.pow((img[h, w][0] / 255.0), lam)
    return img


# M: test algorithms.
fileName = '../Images_Contrast_Test/test1/3-1.png'
img = cv2.imread(fileName, 1)
cv2.imshow('Original Image', img)
img1 = img.copy()
img2 = img.copy()
img3 = img.copy()
adjustedImage1 = mGlobalGammaAdjuster(img1, 2)
adjustedImage2 = mLocalGammaAdjuster_GaussianMask(img2)
adjustedImage3 = mLocalGammaAdjuster_BilateralFilterMask(img3)
cv2.imshow('Normal Gamma Correction', adjustedImage1)
cv2.imshow('Moroney Suggested Correction', adjustedImage2)
cv2.imshow('LCC Correction', adjustedImage3)
cv2.waitKey(0)

