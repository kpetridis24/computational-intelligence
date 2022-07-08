from keras.datasets import mnist
import matplotlib.pyplot as plt


def load_digit_figures():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


def reshape_dataset(X_train, X_test):
    X_train = X_train.reshape((X_train.shape[1] * X_train.shape[2], X_train.shape[0]))
    X_test = X_test.reshape((X_test.shape[1] * X_test.shape[2], X_test.shape[0]))
    return X_train, X_test


def visualize(data):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(data[i], cmap=plt.get_cmap('gray'))
    plt.show()
