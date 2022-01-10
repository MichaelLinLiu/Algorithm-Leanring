import cv2
import numpy as np


def downSampler(image, scale_rate):
    height = image.shape[0] + 1
    width = image.shape[1] + 1
    downSampled_height = height // scale_rate
    downSampled_width = width // scale_rate
    result = np.zeros((downSampled_height, downSampled_width), dtype=np.uint8)

    for h in range(0, height, scale_rate):
        for w in range(0, width, scale_rate):
            i = h // scale_rate
            j = w // scale_rate
            if i < downSampled_height and j < downSampled_width:
                result[i][j] = image[h-1][w-1]
    return result


def bilinear_up_sampler(image, scale_rate):

    # M: read the row of the image
    height = image.shape[0]
    width = image.shape[1]
    mArr = []
    mArr2 = []
    mArr3 = []
    numberOFInterpolation = scale_rate - 1
    result_height = height * scale_rate - (scale_rate - 1)
    result_width = width * scale_rate - (scale_rate - 1)
    result = np.zeros((result_width, result_height), dtype=np.uint8)
    result2 = np.zeros((result_height, result_width), dtype=np.uint8)
    result_row = np.zeros((height, result_width), dtype=np.uint8)
    result_col = np.zeros((result_width, height), dtype=np.uint8)

    for h in range(height):
        for w in range(width-1):
            point1 = int(image[h, w])
            point2 = int(image[h, w + 1])
            difference = point2 - point1
            for i in range(0, numberOFInterpolation+1):
                middleValue = point1 + difference / scale_rate * i
                mArr.append(middleValue)
            if w == width-2:
                mArr.append(point2)

    # M: convert 1D array to 2D array
    count = 0
    for h in range(result_row.shape[0]):
        for w in range(result_row.shape[1]):
            result_row[h,w] = mArr[count]
            count = count + 1

    # M: rotate the image
    for h in range(result_row.shape[1]):
        for w in range(result_row.shape[0]):
            count1 = result_width * w + h
            mArr2.append(mArr[count1])

    count3 = 0
    for h in range(result_col.shape[0]):
        for w in range(result_col.shape[1]):
            result_col[h,w] = mArr2[count3]
            count3 = count3 + 1

    for h in range(result_col.shape[0]):
        for w in range(result_col.shape[1] - 1):
            point1 = int(result_col[h,w])
            point2 = int(result_col[h,w+1])
            difference = point2 - point1
            for i in range(0, numberOFInterpolation+1):
                middleValue = point1 + difference / scale_rate * i
                mArr3.append(middleValue)
        mArr3.append((result_col[h, result_col.shape[1]-1]))

    count4 = 0
    for h in range(result.shape[0]):
        for w in range(result.shape[1]):
            result[h, w] = mArr3[count4]
            count4 = count4 + 1

    for h in range(result2.shape[0]):
        for w in range(result2.shape[1]):
            result2[h, w] = result[w, h]

    return result2


fileName = "../Images_Contrast_Test/test2.png"
img = cv2.imread(fileName, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = img.copy()
sampling_rate = 4
downImage = downSampler(image,sampling_rate)
upimage = bilinear_up_sampler(downImage, sampling_rate)
cv2.imshow('Bilinear Up-sampling, sampling rate:{}'.format(sampling_rate), upimage)
cv2.waitKey(0)




