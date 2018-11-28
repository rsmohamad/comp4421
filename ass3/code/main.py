import cv2
import numpy as np
import matplotlib.pyplot as plt
from digit_segmentation import extractDigits


def showImage(img):
    plt.imshow(img, cmap=plt.get_cmap('binary'))
    plt.axis('off')
    plt.show()


def assignment3(input):
    img = cv2.imread(input)
    markedImg, digits = extractDigits(img)

    for digit in digits:
        showImage(digit)

    showImage(markedImg)


if __name__ == "__main__":
    images = ["1.jpg", "2.bmp"]

    for image in images:
        assignment3("../input_images/" + image)