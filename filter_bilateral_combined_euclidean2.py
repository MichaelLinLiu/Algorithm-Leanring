
# M: This program is a practise of using Euclidean Distance inside the formula of Gaussian's
import numpy as np
import cv2
import time


def mBorder(kernel_size,img):
    # M: create an empty uint8 type array
    # M: the size of padded image should be (kernel size - 1) / 2 + img width + (kernel size - 1) / 2
    paddedImage = np.zeros([kernel_size - 1 + img.shape[0], kernel_size - 1 + img.shape[1]], dtype=np.uint8)
    width = paddedImage.shape[1]
    height = paddedImage.shape[0]
    print(paddedImage.shape, 'height:',height,'width:', width, "pad:",int((kernel_size+1)/2))

    # M: pad the four corners of the padded image
    # M: corner1: top-left-corner
    for h in range(0, int((kernel_size+1)/2) ):
        for w in range(0, int((kernel_size+1)/2) ):
            paddedImage[h,w] = img[0][0]

    # M: corner2: top-right-corner
    for h in range(0, int((kernel_size+1)/2) ):
        for w in range(width - int((kernel_size+1)/2), width):
            paddedImage[h,w] = img[0][-1]

    # # M: corner3: down-left-corner
    for h in range( height - int((kernel_size+1)/2),  height):
        for w in range( 0, int((kernel_size+1)/2) ):
            paddedImage[h,w] = img[-1][0]

    # M: corner4: down-right-corner
    for h in range( height - int((kernel_size+1)/2), height  ):
        for w in range( width - int((kernel_size+1)/2), width ):
            paddedImage[h,w] = img[-1][-1]

    # M: border1: top border
    for h in range(0, int((kernel_size-1)/2) ):
        for w in range( int((kernel_size+1)/2), width - int((kernel_size+1)/2)):
            paddedImage[h,w] = img[0][w - int((kernel_size-1)/2)]

    # M: border2: down border
    for h in range(height - int((kernel_size-1)/2), height):
        for w in range( int((kernel_size+1)/2), width - int((kernel_size+1)/2)):
            paddedImage[h,w] = img[-1][w - int((kernel_size-1)/2)]

    # M: border3: left border
    for h in range( int((kernel_size+1)/2), height - int((kernel_size+1)/2)  ):
        for w in range(0, int((kernel_size-1)/2)):
            paddedImage[h,w] = img[h - int((kernel_size-1)/2)][0]

    # M: border4: right border.
    for h in range( int((kernel_size+1)/2), height - int((kernel_size+1)/2)  ):
        for w in range(width -int((kernel_size-1)/2), width):
            paddedImage[h,w] = img[h - int((kernel_size-1)/2)][-1]

    # M: internal
    for h in range( int((kernel_size-1)/2), height - int((kernel_size-1)/2)  ):
        for w in range(int((kernel_size-1)/2), width - int((kernel_size - 1) / 2)):
            paddedImage[h,w] = img[h - int((kernel_size-1)/2)][w - int((kernel_size-1)/2)]

    return paddedImage


def mPadCropper(kernel_size, paddedImage):
    croppedImage = np.zeros([paddedImage.shape[0] - kernel_size + 1, paddedImage.shape[1] - kernel_size + 1], dtype=np.uint8)
    width = paddedImage.shape[1]
    height = paddedImage.shape[0]
    for h in range(int((kernel_size - 1) / 2), height - int((kernel_size - 1) / 2)):
        for w in range(int((kernel_size - 1) / 2), width - int((kernel_size - 1) / 2)):
            croppedImage[h - int((kernel_size - 1) / 2), w - int((kernel_size - 1) / 2)] = paddedImage[h][w]
    return croppedImage


def convolution(mImg, halfWidth, sigmaRange = 10):
    result = mImg.copy()

    fullWidth = halfWidth * 2 + 1

    sigmaSpatial = fullWidth / np.sqrt(8 * np.log(2))

    # M: loop through the image(not deal with the border yet)
    for h in range(mImg.shape[0] - halfWidth - halfWidth):
        for w in range(mImg.shape[1] - halfWidth - halfWidth):
            roi = mImg[h: h + halfWidth + halfWidth + 1, w: w + halfWidth + halfWidth + 1]
            centerX = halfWidth
            centerY = halfWidth
            centerValue = float(roi[centerY][centerX])
            totalValue = 0.0
            totalWeight = 0.0
            for i in range(len(roi)):
                for j in range(len(roi[0])):
                    q = float(roi[i][j])
                    vector1: float = -0.5 * (centerValue - q) ** 2 / sigmaRange ** 2  # M: this line can be replaced by integral image
                    vector2: float = -0.5 * ((centerX - i) ** 2 + (centerY - j) ** 2) / sigmaSpatial ** 2
                    weightCombined: float = np.exp(vector1 + vector2)
                    weightedValue: float = q * weightCombined
                    totalValue: float = totalValue + weightedValue
                    totalWeight: float = totalWeight + weightCombined

            result[h+halfWidth, w+halfWidth] = totalValue / totalWeight

    return result


def mBilateralBlur(image, kernel_size):
    paddedImage = mBorder(kernel_size,image)
    halfKernelSize = int((kernel_size - 1) / 2)
    convolvedImage = convolution(paddedImage, halfKernelSize)
    croppedImage = mPadCropper(kernel_size, convolvedImage)
    return croppedImage


# M: test the algorithm
startTime = time.perf_counter()
fileName = "../Images_Contrast_Test/test2.png"
img = cv2.imread(fileName, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
halfKernelWidth = 9
kernelSize = halfKernelWidth * 2 + 1
adjustedImage1 = mBilateralBlur(img1, kernelSize)
endTime = time.perf_counter()
print("M: The process took", f"{endTime - startTime:0.4f} Seconds.")
cv2.imshow('mBil vs original, Std:{}'.format(halfKernelWidth), np.hstack((adjustedImage1, img)))
cv2.waitKey(0)