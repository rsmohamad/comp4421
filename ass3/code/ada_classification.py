import numpy as np
import math
import pickle
import cv2

# Utils
from sklearn.metrics import accuracy_score
import utils

# Classifiers
from sklearn import tree
from sklearn import svm
from sklearn import linear_model

classifiers = []
classifierWeights = []
numClassifiers = 5
X_train, X_test, Y_train, Y_test = utils.loadMNIST(test=0.15)
loaded = False

modelFname = 'adaboost_pretrained_best_dtree'


def predict(x, preprocess=True):
    outputs = list(map(lambda clf: clf.predict([x]), classifiers))
    values = list(map(lambda i: np.sum(np.multiply(classifierWeights, np.equal(outputs, i))), range(10)))
    return np.argmax(values)


def train():

    global classifiers, classifierWeights

    sampleWeights = np.ones((len(Y_train))) / len(Y_train)
    classifierWeights = np.zeros(numClassifiers)

    classifiers = [tree.DecisionTreeClassifier(criterion="gini", max_depth=64, max_features='auto') for _ in range(numClassifiers)]
    #classifiers = [svm.SVC(kernel='poly', C=5, gamma=0.05, verbose=1) for _ in range(numClassifiers)]
    #classifiers = [linear_model.logistic.LogisticRegression(multi_class='multinomial', penalty='l1', solver='saga', tol=0.1, verbose=1, n_jobs=4) for _ in range(numClassifiers)]

    for i in range(numClassifiers):
        print("Training classifier %d" % (i+1))

        classifiers[i].fit(X_train, Y_train, sample_weight=sampleWeights)
        Y_hat = classifiers[i].predict(X_train)
        err = 1 - accuracy_score(Y_train, Y_hat, sample_weight=sampleWeights)

        print("Error: %f" % (err))

        classifierWeights[i] = math.log(min((1-err)/err, math.exp(20))) + math.log(9)

        weightUpdate = classifierWeights[i] * np.not_equal(Y_hat, Y_train)
        weightUpdate = np.exp(weightUpdate)
        sampleWeights = np.multiply(sampleWeights, weightUpdate)

        normFactor = np.sum(sampleWeights)
        sampleWeights = sampleWeights/normFactor

    print("Classifier weights:")
    print(classifierWeights)

    test(verbose=True)


def test(verbose=False):
    # Test Classifier
    Y_hat = np.apply_along_axis(func1d=predict, axis=1, arr=X_test)

    if verbose:
        print("Test accuracy: %f" % (accuracy_score(Y_test, Y_hat)))
        utils.showDigits(X_test, Y_hat)


def saveModel(fname):
    obj = [numClassifiers, classifiers, classifierWeights]
    with open(fname, 'wb') as file:
        pickle.dump(obj, file)


def loadModel(fname):
    global classifiers, classifierWeights, numClassifiers, loaded

    if loaded:
        return True

    try:
        with open(fname, 'rb') as file:
            obj = pickle.load(file)
            numClassifiers = obj[0]
            classifiers = obj[1]
            classifierWeights = obj[2]

            test()

    except:
        print("Model not found/loaded properly, training new model")
        return False

    loaded = True
    return True


def adaboostClassification(digits):

    if not loadModel(modelFname):
        train()
        saveModel(modelFname)

    labels = np.apply_along_axis(func1d=predict, axis=1, arr=digits)
    return labels



if __name__ == '__main__':
    train()
    saveModel(modelFname)
