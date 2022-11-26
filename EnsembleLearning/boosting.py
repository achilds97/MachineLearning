import math
import random
import numpy as np
import DecisionTree.id3 as id3
import matplotlib.pyplot as plt


def update_weights(root_node, weights, data):
    new_weights = [0 for i in range(len(weights))]
    alpha = get_alpha(root_node, weights, data)
    for i in range(len(weights)):
        new_weights[i] = weights[i] * math.exp(-1 * alpha * ((1 if data[i]['label'] == '1' else -1) *
                                                             (1 if id3.get_result(root_node, data[i], ['1', '0']) == '1' else -1)))

    new_weights = np.array(new_weights)
    new_weights = (new_weights / sum(new_weights))
    return new_weights, alpha


def get_alpha(root_node, weights, data):
    error = 0
    for i in range(len(data)):
        result = id3.get_result(root_node, data[i], ['1', '0'])
        if result != data[i]['label']:
            error += weights[i]

    result = 0.5 * math.log((1 - error) / error)
    return result


def get_boost_result(boost, example, alpha_set):
    results = []
    for i in range(len(boost)):
        results.append((1 if id3.get_result(boost[i], example, ['1', '0']) == '1' else -1) * alpha_set[i])
    sum_results = sum(results)
    if sum_results > 0:
        return '1'
    else:
        return '0'


def adaboost():
    labels = ['yes', 'no']
    attributes = {'age': ['low', 'high'],
                  'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student',
                          'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
                  'marital': ['married', 'divorced', 'single'],
                  'education': ['unknown', 'secondary', 'primary', 'tertiary'],
                  'default': ['yes', 'no'],
                  'balance': ['low', 'high'],
                  'housing': ['yes', 'no'],
                  'loan': ['yes', 'no'],
                  'contact': ['unknown', 'telephone', 'cellular'],
                  'day': ['low', 'high'],
                  'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                  'duration': ['low', 'high'],
                  'campaign': ['low', 'high'],
                  'pdays': ['low', 'high'],
                  'previous': ['low', 'high'],
                  'poutcome': ['unknown', 'other', 'failure', 'success'],
                  }

    attribute_headers = list(attributes.keys())
    attribute_headers.append('label')

    file = open('../DecisionTree/bank_data/train.csv', 'r')

    examples = []
    for l in file:
        temp = {}
        split_line = l.split(',')
        for i in range(len(split_line)):
            temp[attribute_headers[i]] = split_line[i].strip()
        temp['value'] = 1
        examples.append(temp)

    id3.convert_to_categorical_median(examples, 'age')
    id3.convert_to_categorical_median(examples, 'balance')
    id3.convert_to_categorical_median(examples, 'day')
    id3.convert_to_categorical_median(examples, 'duration')
    id3.convert_to_categorical_median(examples, 'campaign')
    id3.convert_to_categorical_median(examples, 'pdays')
    id3.convert_to_categorical_median(examples, 'previous')

    file = open('../DecisionTree/bank_data/test.csv', 'r')

    test_data = []
    for l in file:
        temp = {}
        split_line = l.split(',')
        for i in range(len(split_line)):
            temp[attribute_headers[i]] = split_line[i].strip()
        temp['value'] = 1
        test_data.append(temp)

    id3.convert_to_categorical_median(test_data, 'age')
    id3.convert_to_categorical_median(test_data, 'balance')
    id3.convert_to_categorical_median(test_data, 'day')
    id3.convert_to_categorical_median(test_data, 'duration')
    id3.convert_to_categorical_median(test_data, 'campaign')
    id3.convert_to_categorical_median(test_data, 'pdays')
    id3.convert_to_categorical_median(test_data, 'previous')

    weights = np.array([1 / len(examples) for i in range(len(examples))])
    boost_set = []
    alpha_set = []

    training_errors = []
    test_errors = []

    for i in range(500):
        print(i)
        for j in range(len(examples)):
            examples[j]['value'] = weights[j]

        root_node = id3.id3(examples=examples, labels=labels, attributes=attributes, calc_method='entropy',
                            cur_depth=0, max_depth=1)
        boost_set.append(root_node)

        weights, alpha = update_weights(root_node, weights, examples)
        alpha_set.append(alpha)

        for i in boost_set:
            pass
            # print(id3.print_tree(i))

        all_results = []
        missed_data = 0
        for i in examples:
            boost_result = get_boost_result(boost_set, i, alpha_set)
            all_results.append([boost_result, i['label']])
            if boost_result != i['label']:
                missed_data += 1

        training_errors.append(missed_data/len(all_results))

        for i in test_data:
            boost_result = get_boost_result(boost_set, i, alpha_set)
            all_results.append([boost_result, i['label']])
            if boost_result != i['label']:
                missed_data += 1

        test_errors.append(missed_data/len(all_results))

    for i in range(500):
        print(i, '\t', training_errors[i], '\t', test_errors[i])

    plt.plot(range(500), training_errors, label='Training Errors')
    plt.plot(range(500), test_errors, label='Testing Errors')

    plt.legend()
    plt.show()


def bagging():
    labels = ['yes', 'no']
    attributes = {'age': ['low', 'high'],
                  'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student',
                          'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
                  'marital': ['married', 'divorced', 'single'],
                  'education': ['unknown', 'secondary', 'primary', 'tertiary'],
                  'default': ['yes', 'no'],
                  'balance': ['low', 'high'],
                  'housing': ['yes', 'no'],
                  'loan': ['yes', 'no'],
                  'contact': ['unknown', 'telephone', 'cellular'],
                  'day': ['low', 'high'],
                  'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                  'duration': ['low', 'high'],
                  'campaign': ['low', 'high'],
                  'pdays': ['low', 'high'],
                  'previous': ['low', 'high'],
                  'poutcome': ['unknown', 'other', 'failure', 'success'],
                  }

    attribute_headers = list(attributes.keys())
    attribute_headers.append('label')

    file = open('../DecisionTree/bank_data/train.csv', 'r')

    examples = []
    for l in file:
        temp = {}
        split_line = l.split(',')
        for i in range(len(split_line)):
            temp[attribute_headers[i]] = split_line[i].strip()
        temp['value'] = 1
        examples.append(temp)

    id3.convert_to_categorical_median(examples, 'age')
    id3.convert_to_categorical_median(examples, 'balance')
    id3.convert_to_categorical_median(examples, 'day')
    id3.convert_to_categorical_median(examples, 'duration')
    id3.convert_to_categorical_median(examples, 'campaign')
    id3.convert_to_categorical_median(examples, 'pdays')
    id3.convert_to_categorical_median(examples, 'previous')

    file = open('../DecisionTree/bank_data/test.csv', 'r')

    test_data = []
    for l in file:
        temp = {}
        split_line = l.split(',')
        for i in range(len(split_line)):
            temp[attribute_headers[i]] = split_line[i].strip()
        temp['value'] = 1
        test_data.append(temp)

    id3.convert_to_categorical_median(test_data, 'age')
    id3.convert_to_categorical_median(test_data, 'balance')
    id3.convert_to_categorical_median(test_data, 'day')
    id3.convert_to_categorical_median(test_data, 'duration')
    id3.convert_to_categorical_median(test_data, 'campaign')
    id3.convert_to_categorical_median(test_data, 'pdays')
    id3.convert_to_categorical_median(test_data, 'previous')

    iterations = 500
    data_per_set = len(examples) / iterations
    split_data = []

    working_set = list(examples)
    for i in range(iterations):
        cur_data = []
        for j in range(int(data_per_set)):
            index = random.randint(0, len(working_set)-1)
            cur_data.append(working_set.pop(index))

        split_data.append(cur_data)

    bagged_trees = []

    for data in split_data:
        bagged_trees.append(id3.id3(data, attributes, labels, 'entropy', 0, 1000000))

    all_results = []
    missed_data = 0
    alpha_set = [1 for i in range(len(bagged_trees))]
    for i in examples:
        boost_result = get_boost_result(bagged_trees, i, alpha_set)
        all_results.append([boost_result, i['label']])
        if boost_result != i['label']:
            missed_data += 1

    print(missed_data / len(all_results))



if __name__ == '__main__':
    adaboost()
