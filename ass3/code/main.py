import cv2
import numpy as np
import matplotlib.pyplot as plt
from digit_segmentation import digitSegmentation


def showImage(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()



def adaboostClassification():
    pass


def assignment3(input):
    img = cv2.imread(input)
    markedImg, digits = digitSegmentation(img)

    # for digit in digits:
    #     showImage(digit)

    showImage(markedImg)


if __name__ == "__main__":
    images = ["1.jpg", "2.bmp"]

    for image in images:
        assignment3("../input_images/" + image)