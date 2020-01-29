import sklearn.datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def load_data():
    breast_cancer = sklearn.datasets.load_breast_cancer()
    data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    data['class'] = breast_cancer.target
    data.head()
    data.describe()
    print(data['class'].value_counts())
    print(breast_cancer.target_names)
    data.groupby('class').mean()
    x = data.drop('class', axis=1)
    y = data['class']
    return x, y, data


def start():
    # 1: load data set completely
    x, y, data = load_data()

    # 2: split data set to train and test
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)
    print(X_train.mean(), X_test.mean(), x.mean())
    X_train = X_train.values
    X_test = X_test.values

    # 3: create an object from class perceptron
    perceptron = Perceptron()

    # 4: calculate weight matrix with 10000 epochs and learning rate 0.3
    wt_matrix = perceptron.fit(X_train, Y_train, 10000, 0.3)

    # 5: calculate weight matrix with 10000 epochs and learning rate 0.5
    wt_matrix = perceptron.fit(X_train, Y_train, 10000, 0.5)

    # 6: predict test set
    Y_pred_test = perceptron.predict(X_test)

    # 7: accuracy
    print(accuracy_score(Y_pred_test, Y_test))
    plt.plot(wt_matrix[-1, :])
    plt.show()


class Perceptron:

    def __init__(self):
        self.w = None
        self.b = None

    def model(self, x):
        return 1 if (np.dot(self.w, x) >= self.b) else 0

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, X, Y, epochs=1, lr=1):

        self.w = np.ones(X.shape[1])
        self.b = 0

        accuracy = {}
        max_accuracy = 0

        wt_matrix = []

        for i in range(epochs):
            if i % 1000 == 0:
                print("epoch: " + str(i))
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b - lr * 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b + lr * 1

            wt_matrix.append(self.w)

            temp = self.predict(X)
            accuracy[i] = accuracy_score(temp, Y)
            if (accuracy[i] > max_accuracy):
                max_accuracy = accuracy[i]
                j = i
                chkptw = self.w
                chkptb = self.b

        self.w = chkptw
        self.b = chkptb

        print(max_accuracy, j)

        plt.plot(list(accuracy.values()))
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.ylim([0, 1])
        plt.show()

        return np.array(wt_matrix)
