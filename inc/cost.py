import numpy as np


def cost_function(y_model, y_target, method="mse"):
    return sum((y_target[i, j] - y_model[i, j]) ** 2 for i in range(y_model.shape[0]) for j in range(y_model.shape[1]))
