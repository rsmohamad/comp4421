import numpy as np
import joblib

# Utils
from sklearn.metrics import accuracy_score
from mnist import load, showDigits

# Classifiers
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load(test=0.15)

    print("Training SVM")
    polySVM = svm.SVC(kernel='poly', C=5, gamma=0.05)
    polySVM.fit(X_train, Y_train)
    Y_hat = polySVM.predict(X_test)
    Y_hat = np.round(Y_hat).astype(int)
    showDigits(X_test, Y_hat, 12, "SVM {}")
    print("SVM: %f" % (accuracy_score(Y_test, Y_hat)))

    print("Training decision tree")
    dTree = tree.DecisionTreeClassifier(criterion="gini", max_depth=32, max_features=784)
    dTree.fit(X_train, Y_train)
    Y_hat = dTree.predict(X_test)
    Y_hat = np.round(Y_hat).astype(int)
    showDigits(X_test, Y_hat, 12, "DT {}")
    print("DT: %f" % (accuracy_score(Y_test, Y_hat)))

    print("Training random forest")
    randForest = ensemble.RandomForestClassifier(n_estimators=150, criterion="gini", max_depth=32, max_features="auto")
    randForest.fit(X_train, Y_train)
    Y_hat = randForest.predict(X_test)
    Y_hat = np.round(Y_hat).astype(int)
    showDigits(X_test, Y_hat, 12, "RF {}")
    print("RF: %f" % (accuracy_score(Y_test, Y_hat)))

    print("Training logistic regression")
    regression = linear_model.logistic.LogisticRegression(multi_class='multinomial', penalty='l1', solver='saga', tol=0.1)
    regression.fit(X_train, Y_train)
    Y_hat = regression.predict(X_test)
    Y_hat = np.round(Y_hat).astype(int)
    showDigits(X_test, Y_hat, 12, "LR {}")
    print("LR: %f" % (accuracy_score(Y_test, Y_hat)))

    joblib.dump(polySVM, "svm.joblib")
    joblib.dump(dTree, "dTree.joblib")
    joblib.dump(randForest, "randForest.joblib")
    joblib.dump(regression, "logRegression.joblib")
