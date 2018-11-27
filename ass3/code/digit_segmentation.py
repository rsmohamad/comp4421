import cv2
import numpy as np


def drawBoundingRect(img, rect):
    rStart = rect[0]
    rEnd = rect[1]
    cStart = rect[2]
    cEnd = rect[3]

    points = [(cStart, rStart), (cEnd, rStart), (cEnd, rEnd), (cStart, rEnd)]

    for i in range(len(points)):
        cv2.line(img, points[i], points[(i+1)%len(points)], (255, 0, 0), 4)


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
        pad = min(abs(cStart-cEnd), abs(rStart-rEnd))/10
        pad = int(pad)

        rect = [rStart-pad, rEnd+pad, cStart-pad, cEnd+pad]
        bbox.append(rect)

    return bbox


def digitSegmentation(img):

    # Get the binary image of digits
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.blur(imgGray, (5, 5))

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    bgImg = cv2.dilate(imgGray, se)
    bgImg = cv2.medianBlur(bgImg, 21)

    textImg = 255 - cv2.absdiff(bgImg, imgGray)
    textImg_cp = textImg.copy()

    cv2.normalize(textImg_cp, textImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, imgBw = cv2.threshold(textImg_cp,0,  255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    digitImgs = []
    markedImg = img.copy()
    rows = divideToRows(imgBw)
    for rStart, rEnd in rows:
        bboxes = divideToCols(imgBw, rStart, rEnd)
        for rect in bboxes:
            digit = img[rect[0]:rect[1], rect[2]:rect[3]]
            drawBoundingRect(markedImg, rect)
            digitImgs.append(digit)

    return markedImg, digitImgs

