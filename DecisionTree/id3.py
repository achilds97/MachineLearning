import math


class node():
    def __init__(self, attribute):
        self.attribute = attribute
        self.branches = []

    def __str__(self):
        return self.attribute

    def add_branch(self, branch_node):
        self.branches.append(branch_node)

    def get_branches_string(self):
        string_branches = []
        for i in self.branches:
            string_branches.append(str(i))
        return string_branches


def print_tree(root_node):
    if len(root_node.branches) > 0:
        print("{} branches: {}".format(root_node, root_node.get_branches_string()))
        for i in root_node.branches:
            print_tree(i)


def id3(examples, attributes, labels):
    if get_entropy(examples, labels) == 0:
        return node(examples[0]['label'])
    if len(attributes) == 0:
        return node(get_common_labels(examples))

    a = get_max_gain(examples, attributes, labels)

    root = node(a)

    cur_attributes = attributes[a]
    del attributes[a]

    for v in cur_attributes:
        value_filter = filter_list(examples, v, a)

        if len(value_filter) == 0 and len(examples) > 0:
            root.branches.append(node(get_common_labels(examples)))
        else:
            ready_data = remove_attribute(value_filter, a)
            if len(examples) > 0:
                root.branches.append(id3(ready_data, attributes, labels))

    return root


def get_entropy(examples, labels):
    cur_entropy = 0

    for l in labels:
        label_count = 0
        for i in examples:
            if i['label'] == l:
                label_count += 1

        if label_count > 0:
            cur_entropy += -(label_count / len(examples)) * math.log2(label_count / len(examples))

    return cur_entropy


def get_gini(examples, labels):
    cur_gini = 0

    for l in labels:
        label_count = 0
        for i in examples:
            if i['label'] == l:
                label_count += 1

        if label_count > 0:
            cur_gini += (label_count / len(examples))**2

    return 1 - cur_gini


def get_max_gain_gini(examples, attributes, labels):
    gain = {}
    total_gini = get_gini(examples, labels)

    for a in attributes:
        cur_gain = 0
        for v in attributes[a]:
            filtered_list = filter_list(examples, v, a)
            gini = get_gini(filtered_list, labels)
            cur_gain += ((len(filtered_list) / len(examples)) * gini)

        gain[a] = total_gini - cur_gain
        print('{}: {}'.format(a, gain[a]))

    return max(gain, key=gain.get)


def get_max_gain(examples, attributes, labels):
    gain = {}
    total_entropy = get_entropy(examples, labels)

    for a in attributes:
        cur_gain = 0
        for v in attributes[a]:
            filtered_list = filter_list(examples, v, a)
            entropy = get_entropy(filtered_list, labels)
            cur_gain += ((len(filtered_list) / len(examples)) * entropy)

        gain[a] = total_entropy - cur_gain
        print('{}: {}'.format(a, gain[a]))


    return max(gain, key=gain.get)


def filter_list(items, filter_item, filter_column):
    filtered_items = []

    for i in items:
        if i[filter_column] == filter_item:
            filtered_items.append(i)

    return filtered_items


def remove_attribute(items, attribute):
    new_items = []
    for i in items:
        new_dict = dict(i)
        del new_dict[attribute]
        new_items.append(new_dict)

    return new_items


def get_common_labels(items):
    labels = {}

    for i in items:
        if i['label'] in labels:
            labels[i['label']] += 1
        else:
            labels[i['label']] = 1

    return max(labels, key=labels.get)


def main():
    examples = [['sunny', 'hot', 'high', 'weak', 'no'],
                ['sunny', 'hot', 'high', 'strong', 'no'],
                ['overcast', 'hot', 'high', 'weak', 'yes'],
                ['rain', 'mild', 'high', 'weak', 'yes'],
                ['rain', 'cool', 'normal', 'weak', 'yes'],
                ['rain', 'cool', 'normal', 'strong', 'no'],
                ['overcast', 'cool', 'normal', 'strong', 'yes'],
                ['sunny', 'mild', 'high', 'weak', 'no'],
                ['sunny', 'cool', 'normal', 'weak', 'yes'],
                ['rain', 'mild', 'normal', 'weak', 'yes'],
                ['sunny', 'mild', 'normal', 'strong', 'yes'],
                ['overcast', 'mild', 'high', 'strong', 'yes'],
                ['overcast', 'hot', 'normal', 'weak', 'yes'],
                ['rain', 'mild', 'high', 'strong', 'no']]

    attribute_strings = ['outlook', 'temperature', 'humidity', 'wind', 'label']

    examples_dict = []
    for i in examples:
        temp = {}
        for j in range(len(i)):
            temp[attribute_strings[j]] = i[j]

        examples_dict.append(temp)

    labels = ['yes', 'no']

    attributes = {'outlook': ['sunny', 'overcast', 'rain'],
                  'temperature': ['cool', 'mild', 'hot'],
                  'humidity': ['high', 'normal'],
                  'wind': ['weak', 'strong']}

    root_node = id3(examples_dict, attributes, labels)

    print_tree(root_node)


def test():
    examples = [['sunny', 'hot', 'high', 'weak', 'no'],
                ['sunny', 'hot', 'high', 'strong', 'no'],
                ['overcast', 'hot', 'high', 'weak', 'yes'],
                ['rain', 'mild', 'high', 'weak', 'yes'],
                ['rain', 'cool', 'normal', 'weak', 'yes'],
                ['rain', 'cool', 'normal', 'strong', 'no'],
                ['overcast', 'cool', 'normal', 'strong', 'yes'],
                ['sunny', 'mild', 'high', 'weak', 'no'],
                ['sunny', 'cool', 'normal', 'weak', 'yes'],
                ['rain', 'mild', 'normal', 'weak', 'yes'],
                ['sunny', 'mild', 'normal', 'strong', 'yes'],
                ['overcast', 'mild', 'high', 'strong', 'yes'],
                ['overcast', 'hot', 'normal', 'weak', 'yes'],
                ['rain', 'mild', 'high', 'strong', 'no']]

    attribute_strings = ['outlook', 'temperature', 'humidity', 'wind', 'label']

    examples_dict = []
    for i in examples:
        temp = {}
        for j in range(len(i)):
            temp[attribute_strings[j]] = i[j]

        examples_dict.append(temp)

    labels = ['yes', 'no']

    attributes = {'outlook': ['sunny', 'overcast', 'rain'],
                  'temperature': ['cool', 'mild', 'hot'],
                  'humidity': ['high', 'normal'],
                  'wind': ['weak', 'strong', 'normal']}

    # print(get_max_gain(examples_dict, attributes, labels))
    # print(get_entropy(filter_list(examples_dict, 'overcast', 'outlook'), labels))
    # print(get_entropy(filter_list(examples_dict, 'rain', 'outlook'), labels))
    # print(get_entropy(filter_list(examples_dict, 'sunny', 'outlook'), labels))

    # rain_filter = filter_list(examples_dict, 'rain', 'outlook')
    # remove_attribute(rain_filter, 'outlook')

    #print(get_max_gain(rain_filter, attributes, labels))

    sunny_examples = [['sunny', 'hot', 'high', 'weak', 'no'],
                      ['sunny', 'hot', 'high', 'strong', 'no'],
                      ['sunny', 'mild', 'high', 'weak', 'no'],
                      ['sunny', 'cool', 'normal', 'weak', 'yes'],
                      ['sunny', 'mild', 'normal', 'strong', 'yes']]

    sunny_attributes = {'temperature': ['cool', 'mild', 'hot'],
                        'humidity': ['high', 'normal'],
                        'wind': ['weak', 'strong', 'normal']}

    sun_filter = filter_list(examples_dict, 'sunny', 'outlook')
    sun_filter = remove_attribute(sun_filter, 'outlook')

    # print(sun_filter)
    # print(labels)
    print(get_max_gain(sun_filter, sunny_attributes, labels))

    # print(get_entropy(filter_list(sun_filter, 'high', 'humidity'), labels))
    print(get_entropy(filter_list(sun_filter, 'mild', 'temperature'), labels))

def hw_1_help():
    examples = [{'x1': '0', 'x2': '0', 'x3': '1', 'x4': '0', 'label': '0'},
                {'x1': '0', 'x2': '1', 'x3': '0', 'x4': '0', 'label': '0'},
                {'x1': '0', 'x2': '0', 'x3': '1', 'x4': '1', 'label': '1'},
                {'x1': '1', 'x2': '0', 'x3': '0', 'x4': '1', 'label': '1'},
                {'x1': '0', 'x2': '1', 'x3': '1', 'x4': '0', 'label': '0'},
                {'x1': '1', 'x2': '1', 'x3': '0', 'x4': '0', 'label': '0'},
                {'x1': '0', 'x2': '1', 'x3': '0', 'x4': '1', 'label': '0'}]

    attributes = {'x1': ['0', '1'], 'x2': ['0', '1'], 'x3': ['0', '1'], 'x4': ['0', '1']}
    labels = ['0', '1']

    #root_node = get_max_gain(examples, attributes, labels)
    #print(root_node)
    #print(get_entropy(examples, labels))
    #root_node = id3(examples, attributes, labels)
    #print(root_node)
    #print_tree(root_node)

    max_gain = get_max_gain(filter_list(examples, '0', 'x2'), attributes, labels)
    print(max_gain)
    print(filter_list(examples, '0', 'x2'))

def test2():
    examples = [['sunny', 'hot', 'high', 'weak', 'no'],
                ['sunny', 'hot', 'high', 'strong', 'no'],
                ['overcast', 'hot', 'high', 'weak', 'yes'],
                ['rain', 'mild', 'high', 'weak', 'yes'],
                ['rain', 'cool', 'normal', 'weak', 'yes'],
                ['rain', 'cool', 'normal', 'strong', 'no'],
                ['overcast', 'cool', 'normal', 'strong', 'yes'],
                ['sunny', 'mild', 'high', 'weak', 'no'],
                ['sunny', 'cool', 'normal', 'weak', 'yes'],
                ['rain', 'mild', 'normal', 'weak', 'yes'],
                ['sunny', 'mild', 'normal', 'strong', 'yes'],
                ['overcast', 'mild', 'high', 'strong', 'yes'],
                ['overcast', 'hot', 'normal', 'weak', 'yes'],
                ['rain', 'mild', 'high', 'strong', 'no'],
                ['overcast', 'mild', 'normal', 'weak', 'yes']]

    attribute_strings = ['outlook', 'temperature', 'humidity', 'wind', 'label']

    examples_dict = []
    for i in examples:
        temp = {}
        for j in range(len(i)):
            temp[attribute_strings[j]] = i[j]

        examples_dict.append(temp)

    labels = ['yes', 'no']

    attributes = {'outlook': ['sunny', 'overcast', 'rain'],
                  'temperature': ['cool', 'mild', 'hot'],
                  'humidity': ['high', 'normal'],
                  'wind': ['weak', 'strong']}

    filtered = filter_list(examples_dict, 'sunny', 'outlook')
    #print(get_max_gain_gini(examples_dict, attributes, labels))
    print(filtered)
    print(get_gini(filtered, labels))
    print(get_max_gain_gini(examples_dict, attributes, labels))





if __name__ == '__main__':
    test2()
