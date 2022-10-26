import math
import numpy as np
import DecisionTree.id3 as id3


def update_weights(root_node, weights, data):
    new_weights = np.array([0 for i in range(len(weights))])
    for i in range(len(weights)):
        alpha = get_alpha(root_node, weights, data)
        new_weights = weights[i] * math.exp(alpha * 1 if data[i]['label'] == 'yes' else -1 * 1 if id3.get_result(
                root_node, data[i], ['yes', 'no']) == 'yes' else -1)

        print(weights[i], end=' ' if (i+1) % 9 != 0 else '\n')

    weights = weights + (new_weights / sum(weights))

    return weights


def get_alpha(root_node, weights, data):
    error = 0

    for i in range(len(data)):
        result = id3.get_result(root_node, data[i], ['yes', 'no'])
        if result != data[-1]:
            error += weights[i]
    # make sure we don't get a domain error on log2
    if error < 1*10**-15 and error > -1*10**-15:
        return 1
    else:
        print('hey', (1-error)/error)
        result = 0.5 * math.log2((1 - error) / error)
        return result


def get_boost_result(boost, example):
    results = []
    for i in boost:
        results.append(id3.get_result(i, example, ['yes', 'no']))

    return max(results, key=results.count)


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

    examples = examples[0:10]

    weights = np.array([1 / len(examples) for i in range(len(examples))])

    boost_set = []

    for i in range(5):
        for j in range(len(examples)):
            examples[j]['value'] = weights[j]

        root_node = id3.id3(examples=examples, labels=labels, attributes=attributes, weights=weights, calc_method='entropy',
                            cur_depth=0, max_depth=1)
        boost_set.append(root_node)

        update_weights(root_node, weights, examples)

    for i in boost_set:
        print(id3.print_tree(i))
    for i in examples:
        boost_result = get_boost_result(boost_set, i)

        print(boost_result, i['label'])


if __name__ == '__main__':
    adaboost()
