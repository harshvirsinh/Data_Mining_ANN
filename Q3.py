# Baseline MLP for MNIST dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


def load_dataset():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


def start():
    x_train, y_train, x_test, y_test = load_dataset()
    classify_data_by_ANN(x_train, y_train, x_test, y_test)


def flatten_images_to_vector(X_train, X_test):
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    return X_train, X_test, num_pixels


def normalize_data(X_train, X_test):
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    return X_train, X_test


def one_hot_encoding(y_train, y_test):
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    return y_train, y_test, num_classes


# define baseline model
def baseline_model(num_pixels, num_classes):
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def classify_data_by_ANN(X_train, y_train, X_test, y_test):
    # 1: flatten 28*28 images to a 784 vector for each image
    X_train, X_test, num_pixels = flatten_images_to_vector(X_train, X_test)

    # 2: normalize data
    X_train, X_test = normalize_data(X_train, X_test)

    # 3: one hot encoding
    y_train, y_test, num_classes = one_hot_encoding(y_train, y_test)

    # 4: build model
    model = baseline_model(num_pixels, num_classes)

    # 5: Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200,
              verbose=2)
    # 6: Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))