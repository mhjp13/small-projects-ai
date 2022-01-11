import numpy as np

input_vector = np.array([1.66, 1.56])
input_vector_2 = np.array([2, 1.5])
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input, weights, bias):
    layer_1 = np.dot(input, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

if __name__ == "__main__":
    print(f"The prediction is {make_prediction(input_vector_2, weights_1, bias)}")