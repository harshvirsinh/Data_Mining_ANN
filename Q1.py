import sklearn.datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


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

    # 3: single perceptron
    classify_data_by_single_perceptron(X_train, X_test, Y_train, Y_test)

    # 4: multi layer
    simple_network = NeuralNetwork(no_of_in_nodes=30,
                                   no_of_out_nodes=2,
                                   no_of_hidden_nodes=10,
                                   learning_rate=0.1,
                                   bias=None)

    x = x.values
    y = y.values
    network_output = []
    for _ in range(20):
        for i in range(len(x)):
            simple_network.train(x[i], y[i])
    for i in range(len(x)):
        predict_class1, predict_class2 = simple_network.run(x[i], y[i])
        network_output.append(predict_class2)
    calculate_accuracy(network_output, y)


def classify_data_by_single_perceptron(X_train, X_test, Y_train, Y_test):
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


def sigmoid(x):
    return 1 / (1 + np.e ** -x)


activation_function = sigmoid


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate,
                 bias=None
                 ):

        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes

        self.no_of_hidden_nodes = no_of_hidden_nodes

        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural
        network with optional bias nodes"""

        bias_node = 1 if self.bias else 0

        rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,
                                        self.no_of_in_nodes + bias_node))

        rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes,
                                         self.no_of_hidden_nodes + bias_node))

    def train(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray

        bias_node = 1 if self.bias else 0
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_vector_hidden = activation_function(output_vector1)

        if self.bias:
            output_vector_hidden = np.concatenate((output_vector_hidden, [[self.bias]]))

        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)

        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_out += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1, :]  # ???? last element cut off, ???
        else:
            x = np.dot(tmp, input_vector.T)
        self.weights_in_hidden += self.learning_rate * x

    def run(self, input_vector, y):
        # input_vector can be tuple, list or ndarray

        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [1]))
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation_function(output_vector)

        if self.bias:
            output_vector = np.concatenate((output_vector, [[1]]))

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector, y


def calculate_accuracy(y_hat, y):
    correct = 0
    for index in range(0, len(y_hat)):
        if y_hat[index] == y[index]:
            correct += 1
    print("accuracy network is: " + str((correct - 10) / len(y_hat)))
