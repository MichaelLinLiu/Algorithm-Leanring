import numpy as np
import cv2


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


def nearestNeighbour_upSampler(image, scale_rate):
    height = image.shape[0]
    width = image.shape[1]
    result_height = height * scale_rate
    result_width = width * scale_rate
    result = np.zeros((result_height, result_width), dtype=np.uint8)

    for h in range(0, result_height, scale_rate):
        for w in range(0, result_width, scale_rate):
            result[h:(h+1) * scale_rate, w:(w + 1) * scale_rate] = image[h // scale_rate, w // scale_rate]
    return result


fileName = "../Images_Contrast_Test/test2.png"
img = cv2.imread(fileName, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
sampling_rate = 4
down_image = downSampler(img1, sampling_rate)
up_image = nearestNeighbour_upSampler(down_image, sampling_rate)
cv2.imshow('Nearest Neighbour Up-sampling, sampling rate:{}'.format(sampling_rate), up_image)
cv2.waitKey(0)

