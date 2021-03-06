import numpy as np
import matplotlib.pyplot as plt

# Utils
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split


def showImage(img):
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.show()


def showDigits(images, targets, sample_size=24, title_text='Digit {}', random=True, reshape=True):
    nsamples = sample_size

    if random:
        rand_idx = np.random.choice(images.shape[0], nsamples)
    else:
        rand_idx = np.array(range(nsamples))

    images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))

    img = plt.figure(1, figsize=(15, 12), dpi=160)
    for index, (image, label) in enumerate(images_and_labels):
        plt.subplot(np.ceil(nsamples / 6.0), 6, index + 1)
        plt.axis('off')
        # each image is flat, we have to reshape to 2D array 28x28-784

        if reshape:
            plt.imshow(image.reshape(28, 28), cmap=plt.get_cmap('gray'), interpolation='nearest')
        else:
            plt.imshow(image, cmap=plt.get_cmap('gray'), interpolation='nearest')

        plt.title(title_text.format(label))

    plt.show()


def loadMNIST(test=0.15):
    mnist = fetch_mldata('MNIST Original', data_home='./')
    X = mnist.data
    Y = mnist.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test, random_state=69)
    return X_train, X_test, Y_train, Y_test
