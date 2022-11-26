import numpy as np
import random
import scipy.optimize as opt


def svm_subgradient(vals, epochs, initial_learning_rate, c, a):
    weights = [0 for i in range(len(vals[0]['values']))]
    for t in range(epochs):
        # cur_learning_rate = initial_learning_rate / (1 + (initial_learning_rate / a) * t)
        cur_learning_rate = initial_learning_rate / (1 + t)
        random.shuffle(vals)
        for e in vals:
            if e['result'] * np.dot(np.array(e['values']), np.array(weights)) <= 1:
                new_weights = weights - cur_learning_rate * np.array(weights) \
                              + cur_learning_rate * c * len(vals) * e['result'] * np.array(e['values'])
            else:
                new_weights = (1 - cur_learning_rate) * np.array(weights)
            weights = new_weights

    return weights


def svm_dual(alphas, *args):
    cur_sum = 0
    for i in range(len(args[0])):
        for j in range(len(args[0])):
            cur_sum += args[0][i]['result'] * args[0][j]['result'] * \
                       np.dot(args[0][i]['values'], args[0][j]['values']) * alphas[i] * alphas[j] - np.sum(alphas)

    cur_sum = cur_sum * 0.5
    print(cur_sum)
    return cur_sum


def constraint(alpha, *args):
    sum = 0
    for i in range(len(alpha)):
        sum += alpha[i] * args[i]['result']
    return sum


def get_data(file_path):
    file = open(file_path)
    data = []
    for l in file:
        i = list(map(float, l.strip().split(',')))
        temp = {}
        temp['values'] = np.array(i[:-1])
        temp['result'] = i[-1]

        data.append(temp)

    return data


def main():
    train_data = get_data('../perceptron/bank-note/bank-note/train.csv')
    test_data = get_data('../perceptron/bank-note/bank-note/test.csv')

    for c in [100 / 873, 500 / 873, 700 / 873]:
        print('c: ', c)
        weights = svm_subgradient(train_data, 100, 0.01, c, 2)

        error_count = 0
        for e in train_data:
            result = np.dot(e['values'], weights)
            if result * e['result'] < 0:
                error_count += 1

        print(error_count / len(train_data))

        error_count = 0
        for e in test_data:
            result = np.dot(e['values'], weights)
            if result * e['result'] < 0:
                error_count += 1

        print(error_count / len(train_data))

    initial_alphas = [0 for i in range(len(train_data))]
    alphas = opt.minimize(svm_dual, initial_alphas, args=train_data, method='SLSQP',
                          constraints={'type': 'eq', 'fun': constraint, 'args': train_data})


if __name__ == '__main__':
    main()
