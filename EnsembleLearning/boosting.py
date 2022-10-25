import math
import numpy as np
import DecisionTree.id3 as id3


def update_weights(root_node, weights, data):
    for i in range(len(weights)):
        weights[i] = weights[i] * math.exp(
            get_alpha(root_node, weights, data) * data[-1] * 1 if id3.get_result(root_node, data[:-1],
                                                                                 ['yes', 'no']) == 'yes' else -1)

    weights = weights / sum(weights)

    return weights


def get_alpha(root_node, weights, data):
    error = 0

    for i in range(len(data)):
        result = id3.get_result(root_node, data[:-1], ['yes', 'no'])
        if result != data[-1]:
            error += weights[i]
    if error == 0:
        return 1
    else:
        return 0.5 * math.log2((1 - error) / error)

def get_decision_stump(data, weights, labels, attributes):
    root_node = id3.id3(examples=data, labels=labels, attributes=attributes, weights=weights, calc_method='gini',
                        cur_depth=0, max_depth=1)
