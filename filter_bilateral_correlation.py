# M: A very fast Bilateral Filter solution by using correlation

import time
import numpy as np
import cv2


# M: Gaussian formula without integral part
def gaussian_kernel(x, sigma):
    return np.exp(-0.5 * x / sigma ** 2)


def filter_bilateral(img, kernel_size, sigma_v, reg_constant=1e-8):

    sigma_s = kernel_size / np.sqrt(8 * np.log(2))

    wgt_sum = np.ones(img.shape )* reg_constant

    # M: make the precision of image to 0.00000001
    result = img * reg_constant

    for x in range(-kernel_size, kernel_size+1):
        for y in range(-kernel_size, kernel_size+1):

            # M: use simplified Gaussian formula to get spatial weight
            weightSpatial = gaussian_kernel(x ** 2 + y ** 2, sigma_s)

            # shift by the offsets,offset is the reference point.
            shiftImg = np.roll(img, [y, x], axis=[0, 1])

            # M: use Gaussian 1D to get range weight
            weightRange = gaussian_kernel((shiftImg - img) ** 2, sigma_v)
            # print(weightRange)

            # compute the value weight
            totalWeights = weightSpatial * weightRange

            # M: the weights of reference point
            center_totalWeights = totalWeights * shiftImg
            # print(center_totalWeights)

            # accumulate the results
            result += center_totalWeights
            wgt_sum += totalWeights

    # normalize the result and return
    return result / wgt_sum


# M: test algorithms.
fileName = '../Images_Contrast_Test/test2.png'
img = cv2.imread(fileName, 1)
img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

adjustedImage4 = img .astype(np.float32)/255.0

size = 2
startTime = time.perf_counter()
adjustedImage4 = filter_bilateral( adjustedImage4, size, 0.1 )
endTime = time.perf_counter()
print("M: The process took", f"{endTime - startTime:0.4f} Seconds.")
cv2.imshow('bil', adjustedImage4)
cv2.waitKey(0)

