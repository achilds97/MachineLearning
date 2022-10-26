import numpy as np
import matplotlib.pyplot as plt


# This takes one step forward in the gradient descent, using the least mean squares method of cost calculation.
# To do full batch gradient descent, pass in the whole data set as data, to do small batch gradient descent pass in
# whatever batch size you would like to calculate (IE a data subset with 10 rows) For stochastic gradient descent pass
# in a single line from the full dataset.
def step_gradient_descent(data, weights, step_size):
    weights_derivative = np.array([0.0 for x in range(len(weights))])

    for i in data:
        error = i[-1] - (np.dot(i[:-1], weights))
        temp_error = np.array([0.0 for x in range(len(weights))])
        for j in range(len(i) - 1):
            temp_error[j] = error * i[j]
        weights_derivative += temp_error

    return weights + step_size*weights_derivative


# Gets the error using the given weights and a least squared error function on the given dataset. If there is an
# intercept it is expected to be in the first position of the weights vector, with a 1 prepended to the data in the
# dataset.
def get_error(data, weights):
    error = 0

    for i in data:
        error += (i[-1] - (np.dot(weights, i[:-1])))**2

    return error * 0.5


# Runs batch gradient descent based on the given data using the give batch size, number of steps, and step size.
def batch_gradient_descent(data, steps, step_size, batch_size):
    weights = np.array([0 for i in range(len(data[0]) - 1)])

    errors = []
    for i in range(steps):
        weights = step_gradient_descent(data[(i * batch_size) % len(data): ((i+1) * batch_size) % len(data)], weights, step_size)
        errors.append(get_error(data, weights))

    return weights, errors


# To calculate the intercept as well as the weights prepend a 1 to every line of data in your dataset. Then once the
# algorithm has finished you will have the optimal value for the intercept in the first position of the weight vector.
def run_gradient_descent():
    file = open('concrete/train.csv', 'r')

    data = []
    for l in file:
        temp = [1]
        split_line = l.split(',')
        for i in range(len(split_line)):
            temp.append(split_line[i].strip())

        data.append(temp)

    file = open('concrete/test.csv', 'r')

    data = []
    for l in file:
        temp = [1]
        split_line = l.split(',')
        for i in range(len(split_line)):
            temp.append(split_line[i].strip())

        data.append(temp)

    iterations = 500
    train_rate = 0.1
    batch_size = len(data) - 1

    weights, errors = batch_gradient_descent(data, iterations, train_rate, batch_size)

    print(weights)

    x = [i for i in range(len(errors))]
    plt.plot(x, errors)
    plt.show()

    print(errors[-1])


if __name__ == '__main__':
    run_gradient_descent()

