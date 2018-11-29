import cv2
import numpy as np
import math


def preprocessMnist(img):
    largerDim = np.max(img.shape)
    smallerDim = np.min(img.shape)

    cv2.normalize(img.copy(), img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    seSize = int(max(1., smallerDim / 12))
    if largerDim > 28:
        img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (seSize, seSize + 1)))

    newImg = np.zeros((largerDim, largerDim))
    remain = largerDim - smallerDim
    start = math.floor(remain / 2)

    if img.shape[0] > img.shape[1]:
        newImg[:, start:start + smallerDim] = img
    else:
        newImg[start:start + smallerDim, :] = img


    resizedImg = cv2.resize(newImg, (20, 20), interpolation=cv2.INTER_LINEAR)
    resizedImg = cv2.copyMakeBorder(resizedImg, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    nonzero_idx = np.nonzero(resizedImg)
    resizedImg[nonzero_idx] += 3*255
    resizedImg[nonzero_idx] /= 4

    resizedImg = resizedImg.astype('uint8')
    return resizedImg


def drawBoundingRect(img, rect):
    rStart = rect[0]
    rEnd = rect[1]
    cStart = rect[2]
    cEnd = rect[3]

    points = [(cStart, rStart), (cEnd, rStart), (cEnd, rEnd), (cStart, rEnd)]

    for i in range(len(points)):
        cv2.line(img, points[i], points[(i + 1) % len(points)], (255, 0, 0), 2)


# DONT MODIFY
def divideToRows(imgBw):
    seSize = int(min(imgBw.shape[0], imgBw.shape[1]) / 20)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seSize, seSize))
    imgBw = cv2.dilate(imgBw, se)
    imgBw = cv2.erode(imgBw, se)

    yProjection = np.sum(imgBw, axis=1)

    start = 0
    rows = []
    isInside = False
    for i in range(len(yProjection)):
        val = yProjection[i]
        if val > 0 and not isInside:
            start = i
            isInside = True
        elif val <= 0 and isInside:
            isInside = False
            if i - start > len(yProjection) / 15:
                rows.append((start, i))

    return rows

def divideToCols(img, rStart, rEnd):
    yProjection = np.sum(img[rStart:rEnd, :], axis=0)
    start = 0
    cols = []
    isInside = False
    for i in range(len(yProjection)):
        val = yProjection[i]
        if val > 0 and not isInside:
            start = i
            isInside = True
        elif val == 0 and isInside:
            isInside = False
            cols.append((start, i))

    bbox = []
    for cStart, cEnd in cols:
        pad = 5
        rect = [rStart - pad, rEnd + pad, cStart - pad, cEnd + pad]
        bbox.append(rect)

    return bbox


def segmentConsolidatedDigits(imgBw):
    regions, labels = cv2.connectedComponents(imgBw)
    bbox = []
    for i in range(1, regions):
        nonzero_idx = np.nonzero(labels == i)
        rStart = np.min(nonzero_idx[0])
        rEnd = np.max(nonzero_idx[0])
        cStart = np.min(nonzero_idx[1])
        cEnd = np.max(nonzero_idx[1])
        rect = [rStart, rEnd, cStart, cEnd]

        if (rEnd - rStart)/imgBw.shape[0] > 0.1:
            bbox.append(rect)

    return bbox


def extractDigits(img):
    # Get the binary image of digits
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bgImg = cv2.dilate(imgGray, se)
    bgImg = cv2.blur(bgImg, (2, 2))
    bgImg = cv2.medianBlur(bgImg, 21)

    textImg = 255 - cv2.absdiff(bgImg, imgGray)
    textImg_cpy = textImg.copy()
    cv2.normalize(textImg_cpy, textImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    imgBw1 = cv2.adaptiveThreshold(textImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 90)
    _, imgBw2 = cv2.threshold(textImg, 90, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    imgBw = np.bitwise_and(imgBw1, imgBw2)

    seSize = int(max(2., np.max(imgBw.shape) / 100))

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (seSize, 5))
    imgBw = cv2.dilate(imgBw, se)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    imgBw = cv2.dilate(imgBw, se)
    imgBw = cv2.erode(imgBw, se)

    digitImgs = np.empty(0)
    markedImg = img.copy()
    rows = divideToRows(imgBw.copy())
    for rStart, rEnd in rows:
        bboxes = divideToCols(imgBw, rStart, rEnd)
        for rect in bboxes:
            digit = textImg[rect[0]:rect[1], rect[2]:rect[3]]
            digitBw = imgBw[rect[0]:rect[1], rect[2]:rect[3]]

            bboxes2 = segmentConsolidatedDigits(digitBw)
            for rect2 in bboxes2:
                digit2 = digit[rect2[0]:rect2[1], rect2[2]:rect2[3]]
                digit2 = preprocessMnist(digit2)
                fk = (rect[0] + rect2[0], rect[0]+rect2[1], rect[2] + rect2[2], rect[2] + rect2[3])
                drawBoundingRect(markedImg, fk)
                digitImgs = np.append(digitImgs, digit2)

    digitImgs = np.reshape(digitImgs, (-1, 784))

    return markedImg, digitImgs
