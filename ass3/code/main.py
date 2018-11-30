import cv2
import matplotlib.pyplot as plt
from digit_segmentation import extractDigits
from ada_classification import adaboostClassification
import utils
import numpy as np

def assignment3(input):
    img = cv2.imread(input)
    markedImg, digits = extractDigits(img)

    labels = adaboostClassification(digits)
    utils.showDigits(digits, labels, labels.shape[0], title_text="label = {}", random=False)
    utils.showImage(markedImg)
    return markedImg


if __name__ == "__main__":
    images = ["1.jpg", "2.bmp", "3.bmp"]

    for i in range(3):
        assignment3("../input_images/" + images[i])
