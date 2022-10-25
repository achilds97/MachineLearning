import numpy as np
import matplotlib.pyplot as plt


def step_gradient_descent(data, weights, intercept, step_size):
    weights_derivative = np.array(weights)
    weights_derivative = np.array([0 for x in range(len(weights))])

    for i in data:
        error = i[-1] - (intercept + np.dot(i[:-1], weights))
        temp_error = np.array([0 for x in range(len(weights))])
        for j in range(len(i) - 1):
            temp_error[j] = error * i[j]

        weights_derivative += temp_error

    return weights + step_size*weights_derivative


def get_error(data, weights, intercept):
    error = 0

    for i in data:
        error += (i[-1] - (intercept + np.dot(weights, i[:-1])))**2

    return error * 0.5


def main():
    data = [np.array([1, -1, 2, 1]),
            np.array([1, 1, 3, 4]),
            np.array([-1, 1, 0, -1]),
            np.array([1, 2, -4, -2]),
            np.array([3, -1, -1, 0])]

    weights = np.array([0.5435172, 0.54315937, 0.89853513])
    weights = np.array([0,0,0])
    intercept = 0.0
    step_size = 0.01

    errors = []
    steps = 100
    print(get_error(data,weights,intercept))
    for i in range(steps):
        weights = step_gradient_descent(data, weights, intercept, step_size)
        errors.append(get_error(data, weights, intercept))
        print(errors[-1], weights)

    print(weights)
    print(errors[-1])
    print(min(errors))

    x = [i for i in range(steps)]
    plt.plot(x, errors)
    plt.show()

if __name__ == '__main__':
    main()

