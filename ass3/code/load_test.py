import pickle
import mnist

# Classifiers
from sklearn import tree

def loadModel(fname):
    with open(fname, 'rb') as file:
        return pickle.load(file)

X_train, X_test, Y_train, Y_test = mnist.load()

svmModel = loadModel("polySvm")
print (svmModel.score(X_test, Y_test))