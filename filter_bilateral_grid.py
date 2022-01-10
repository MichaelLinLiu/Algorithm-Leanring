import math

import cv2
import numpy as np
import time


def image_normalizer(img, new_range, old_range=(0, 255)):
    result_img = np.zeros(img.shape)

    halfKernelWidth = 1
    fullWidth = halfKernelWidth * 2 + 1
    sigmaSpatial = fullWidth / np.sqrt(8 * np.log(2))
    sigmaRange = 10
    mGrid = []
    # grid_element = (0,0,0)
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            vector1 = img[h][w] - old_range[0]
            vector2 = new_range[1] - new_range[0]
            vector3 = old_range[1] - old_range[0]
            normalisedValue = vector1 * vector2 / vector3 + new_range[0]
            # result_img[h][w] = normalisedValue
            divider = normalisedValue/sigmaRange
            if divider == 0:
                grid_element = (0, 0)
            else:
                grid_element = (h/sigmaSpatial/divider, w/sigmaSpatial/divider)

            mGrid.append(grid_element)


    for i in mGrid:
        # vector1 = -0.5 * (p - q) ** 2
        # vector2 = -0.5 * ((i - halfWidth) ** 2 + (j - halfWidth) ** 2)
        # weightCombined = np.exp(vector1 + vector2)
        # weightCombinedKernel[i][j] = weightCombined
        # weightAccumulated = weightAccumulated + weightCombined
        print(i)
    return mGrid





# M: test the algorithm
# fileName = "../Images_Contrast_Test/test6.png"
# img = cv2.imread(fileName, 1)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img1 = img.copy()
# adjustedImage = image_normalizer(img1, (0, 1))

import scipy.signal, scipy.interpolate
def bilateral_approximation(image, sigmaS, sigmaR, samplingS=None, samplingR=None):
    inputHeight = image.shape[0]
    inputWidth = image.shape[1]
    samplingS = sigmaS
    samplingR = sigmaR
    edgeMax = np.amax(image)
    edgeMin = np.amin(image)
    edgeDelta = edgeMax - edgeMin

    derivedSigmaS = sigmaS / samplingS
    derivedSigmaR = sigmaR / samplingR

    paddingXY = math.floor(2*derivedSigmaS)+1
    paddingZ = math.floor(2*derivedSigmaR)+1

    downsampledWidth = int(round((inputWidth-1)/samplingS)+1+2*paddingXY)
    downsampledHeight = int(round((inputHeight-1)/samplingS)+1+2*paddingXY)
    downsampledDepth = int(round(edgeDelta/samplingR)+1+2*paddingZ)

    wi = np.zeros((downsampledHeight,downsampledWidth,downsampledDepth))
    w = np.zeros((downsampledHeight,downsampledWidth,downsampledDepth))

    (ygrid,xgrid) = np.meshgrid(range(inputWidth),range(inputHeight))
    dimx = np.around(xgrid / samplingS)+paddingXY
    dimy = np.around(ygrid / samplingS) + paddingXY
    dimz = np.around((image-edgeMin) / samplingR) + paddingZ

    flat_image = image.flatten()
    flatx = dimx.flatten()
    flaty = dimy.flatten()
    flatz = dimz.flatten()

    for k in range(dimz.size):
        image_k = flat_image[k]
        dimx_k = int(flatx[k])
        dimy_k = int(flaty[k])
        dimz_k = int(flatz[k])

        wi[dimx_k,dimy_k,dimz_k] += image_k
        w[dimx_k, dimy_k, dimz_k] += 1


    kernelWidth = 2* derivedSigmaS + 1
    kernelHeight = kernelWidth
    kernelDepth = 2 * derivedSigmaR + 1

    halfKernelWidth = math.floor(kernelWidth / 2)
    halfKernelHeight = math.floor(kernelHeight / 2)
    halfKernelDepth = math.floor(kernelDepth / 2)

    (gridX,gridY,gridZ) = np.meshgrid(range(int(kernelWidth)),range(int(kernelHeight)),range(int(kernelDepth)) )
    gridX -= halfKernelWidth
    gridY -= halfKernelHeight
    gridZ -= halfKernelDepth
    gridRSquared = ((gridX * gridX + gridY * gridY) / (derivedSigmaS * derivedSigmaS))\
                   + ((gridZ * gridZ) / (derivedSigmaR * derivedSigmaR))
    kernel = np.exp(-0.5 * gridRSquared)
    blurredGridData = scipy.signal.fftconvolve(wi,kernel, mode= 'same')
    blurredGridWeights = scipy.signal.fftconvolve(w,kernel,mode='same')

    # --------------- divide --------------- #
    blurredGridWeights = np.where(blurredGridWeights == 0, -2, blurredGridWeights)
    # avoid divide by 0, won&#39;t read there anyway
    normalizedBlurredGrid = blurredGridData / blurredGridWeights
    normalizedBlurredGrid = np.where(blurredGridWeights < -1, 0, normalizedBlurredGrid)  # put 0s where it&#39;s undefined
    # --------------- 上采样 --------------- #

    (ygrid, xgrid) = np.meshgrid(range(inputWidth), range(inputHeight))

    # 上采样索引
    dimx = (xgrid / samplingS) + paddingXY
    dimy = (ygrid / samplingS) + paddingXY
    dimz = (image - edgeMin) / samplingR + paddingZ

    out_image = scipy.interpolate.interpn((range(normalizedBlurredGrid.shape[0]),
    range(normalizedBlurredGrid.shape[1]),
    range(normalizedBlurredGrid.shape[2])),
    normalizedBlurredGrid,
    (dimx, dimy, dimz))
    return out_image


if __name__ == "__main__":

    image = cv2.imread("../Images_Contrast_Test/test2.png", 0)
    mean_image = bilateral_approximation(image, sigmaS=64, sigmaR=22, samplingS=32, samplingR=16)
    mean_image =np.around(mean_image).astype(np.uint8)
    cv2.imshow('filter',mean_image)
    cv2.waitKey(0)