import numpy as np
import random as rand


def standard_perceptron(data, learning_rate, epochs):
    weights = np.array([0 for i in range(len(data[0]['values']))])

    for t in range(epochs):
        rand.shuffle(data)
        for i in data:
            cur_result = np.dot(weights, i['values'])
            if cur_result * i['result'] <= 0:
                weights = weights + learning_rate * (i['result'] * i['values'])

    return weights


def voted_perceptron(data, learning_rate, epochs):
    weights = np.array([0 for i in range(len(data[0]['values']))])

    num_correct = []
    all_weights = []
    count = 1
    for t in range(epochs):
        rand.shuffle(data)
        for i in data:
            cur_result = np.dot(weights, i['values'])
            if cur_result * i['result'] <= 0:
                all_weights.append(weights)
                num_correct.append(count)
                weights = weights + learning_rate * (i['result'] * i['values'])
                count = 1
            else:
                count += 1

    return num_correct, all_weights


def average_perceptron(data, learning_rate, epochs):
    weights = np.array([0 for i in range(len(data[0]['values']))])
    averages = np.array([0 for i in range(len(data[0]['values']))])

    for t in range(epochs):
        rand.shuffle(data)
        for i in data:
            cur_result = np.dot(weights, i['values'])
            if cur_result * i['result'] <= 0:
                weights = weights + learning_rate * (i['result'] * i['values'])
            averages = averages + weights
    return averages


def main():
    file = open('bank-note/bank-note/train.csv')
    data = []
    for l in file:
        i = list(map(float, l.strip().split(',')))
        temp = {}
        temp['values'] = np.array(i[:-1])
        temp['result'] = -1 if i[-1] == 0 else 1

        data.append(temp)

    file = open('bank-note/bank-note/test.csv')
    test_data = []
    for l in file:
        i = list(map(float, l.strip().split(',')))
        temp = {}
        temp['values'] = np.array(i[:-1])
        temp['result'] = -1 if i[-1] == 0 else 1

        test_data.append(temp)

    weights = standard_perceptron(data, 0.1, 10)
    print(weights)

    missed_count = 0
    for i in data:
        cur_result = np.dot(weights, i['values'])
        if cur_result * i['result'] <= 0:
            missed_count += 1

    print('error rate: ', missed_count / len(data))

    missed_count = 0
    for i in test_data:
        cur_result = np.dot(weights, i['values'])
        if cur_result * i['result'] <= 0:
            missed_count += 1

    print('error rate: ', missed_count / len(test_data))

    correct_counts, weights_list = voted_perceptron(data, 0.1, 10)

    for i in range(len(weights_list)):
        print('$', ["%.5f" % member for member in weights_list[i].tolist()], '\;', correct_counts[i], '$', '\\\\')

    missed_count = 0
    for i in data:
        cur_result = 0
        for x in range(len(weights_list)):
            if np.dot(weights_list[x], i['values']) > 0:
                cur_result += correct_counts[x]
            else:
                cur_result -= correct_counts[x]
        if cur_result * i['result'] <= 0:
            missed_count += 1

    print('error rate: ', missed_count / len(test_data))

    missed_count = 0
    for i in test_data:
        cur_result = 0
        for x in range(len(weights_list)):
            if np.dot(weights_list[x], i['values']) > 0:
                cur_result += correct_counts[x]
            else:
                cur_result -= correct_counts[x]
        if cur_result * i['result'] <= 0:
            missed_count += 1

    print('error rate: ', missed_count / len(test_data))

    weights = average_perceptron(data, 0.1, 10)
    print(weights)

    missed_count = 0
    for i in data:
        cur_result = np.dot(weights, i['values'])
        if cur_result * i['result'] <= 0:
            missed_count += 1

    print('error rate: ', missed_count / len(data))

    missed_count = 0
    for i in test_data:
        cur_result = np.dot(weights, i['values'])
        if cur_result * i['result'] <= 0:
            missed_count += 1

    print('error rate: ', missed_count / len(test_data))



if __name__ == '__main__':
    main()
