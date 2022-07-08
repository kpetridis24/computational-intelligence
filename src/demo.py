import time

import numpy as np
from inc.Helpers.data import load_digit_figures, visualize, reshape_dataset
from inc.activation import sigmoid, ReLU
from inc.cost import cost_function


# Load the dataset
X_train, y_train, X_test, y_test = load_digit_figures()
# visualize(X_train)

# Transform the shape of the dataset from 60kx28x28 to 784x60k
X_train, X_test = reshape_dataset(X_train, X_test)

# print(f"X train: {str(X_train.shape)}")
# print(f"y train: {str(X_train.shape)}")
# print(f"X test: {str(X_test.shape)}")
# print(f"y test: {str(y_test.shape)}")

# Initially use arbitrary weights and biases
# Every row of weight, corresponds to the weights between the first layer and one neuron
weights = np.zeros((X_train.shape[0], X_train.shape[0]))
biases = np.ones((X_train.shape[0],))

# Include biases into the weight matrix, adjust the training set
weights_augmented = np.column_stack((biases, weights))
X_train = np.concatenate((np.ones((1, X_train.shape[1])), X_train))

print(f"X train: {str(X_train.shape)}")
print(f"Weights: {str(weights_augmented.shape)}")

layer2_activation_values = sigmoid(np.matmul(weights_augmented, X_train))
print(layer2_activation_values.shape)
print(layer2_activation_values)
