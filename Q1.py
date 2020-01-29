import sklearn.datasets
import numpy as np


def load_data():
    breast_cancer = sklearn.datasets.load_breast_cancer()
    x = breast_cancer.data
    y = breast_cancer.target
    return x, y


def start():
    x, y = load_data()
