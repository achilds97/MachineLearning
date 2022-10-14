import math


# The base node class for creating a tree.
class Node():
    def __init__(self, attribute):
        self.attribute = attribute
        self.branches = {}

    def __str__(self):
        return self.attribute

    # adds the given node as a branch of this node.
    def add_branch(self, branch_node):
        self.branches.append(branch_node)

    # creates branches from the string so that printing branches doesn't just print a bunch of memory addresses.
    def get_branches_string(self):
        string_branches = []
        for i in self.branches.keys():
            string_branches.append(str(i) + ': ' + str(self.branches[i]))
        return string_branches


# prints the data tree by printing out each node and its branches. Useful for small trees, but becomes quite difficult
# to understand after a certain size of tree.
def print_tree(root_node):
    if len(root_node.branches) > 0:
        print("{} branches: {}".format(root_node, root_node.get_branches_string()))
        for i in root_node.branches:
            print_tree(root_node.branches[i])


# This builds a decision tree for the given dataset using the given calculation method and limits it to the max depth.
def id3(examples, attributes, labels, calc_method, cur_depth, max_depth):
    if cur_depth < max_depth:
        if get_entropy(examples, labels) == 0:
            return Node(examples[0]['label'])
        if len(attributes) == 0:
            return Node(get_common_labels(examples))

        if calc_method == 'entropy':
            a = get_max_gain(examples, attributes, labels)
        elif calc_method == 'gini':
            a = get_max_gain_gini(examples, attributes, labels)
        elif calc_method == 'me':
            a = get_max_gain_me(examples, attributes, labels)

        root = Node(a)

        cur_attributes = attributes[a]
        new_attributes = dict(attributes)
        del new_attributes[a]

        for v in cur_attributes:
            value_filter = filter_list(examples, v, a)

            if len(value_filter) == 0 and len(examples) > 0:
                root.branches[v] = Node(get_common_labels(examples))
            else:
                ready_data = remove_attribute(value_filter, a)
                if len(examples) > 0:
                    root.branches[v] = id3(ready_data, new_attributes, labels, calc_method, cur_depth + 1, max_depth)

        return root
    else:
        return Node(get_common_labels(examples))


# Gets the purity of the set using Entropy
def get_entropy(examples, labels):
    cur_entropy = 0

    sum_examples = sum(item['value'] for item in examples)

    for l in labels:
        label_count = 0
        for i in examples:
            if i['label'] == l:
                label_count += i['value']

        if label_count > 0:
            cur_entropy += -(label_count / sum_examples) * math.log2(label_count / sum_examples)

    return cur_entropy

# Returns the attribute that has the highest information gain in the set based on Entropy method of purity
# calculation.
def get_max_gain(examples, attributes, labels):
    gain = {}
    total_entropy = get_entropy(examples, labels)

    for a in attributes:
        cur_gain = 0
        for v in attributes[a]:
            filtered_list = filter_list(examples, v, a)

            sum_filtered_list = sum(item['value'] for item in filtered_list)
            sum_examples = sum(item['value'] for item in examples)

            entropy = get_entropy(filtered_list, labels)
            cur_gain += ((sum_filtered_list / sum_examples) * entropy)

        gain[a] = total_entropy - cur_gain
    return max(gain, key=gain.get)

# Gets the purity of the set using Gini
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


# Returns the attribute that has the highest information gain in the set based on Gini method of purity
# calculation.
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

    return max(gain, key=gain.get)


# Gets the purity of the set using majority error
def get_me(examples, labels):
    cur_gini = 0
    label_counts = []

    for l in labels:
        label_count = 0
        for i in examples:
            if i['label'] == l:
                label_count += 1

        label_count > 0 and label_counts.append(label_count)

    if len(label_counts) > 0:
        cur_gini += (sum(label_counts) / max(label_counts) / len(examples))

    return 1 - cur_gini


# Returns the attribute that has the highest information gain in the set based on Majority Error method of purity
# calculation.
def get_max_gain_me(examples, attributes, labels):
    gain = {}
    total_me = get_me(examples, labels)

    for a in attributes:
        cur_gain = 0
        for v in attributes[a]:
            filtered_list = filter_list(examples, v, a)
            me = get_me(filtered_list, labels)
            cur_gain += ((len(filtered_list) / len(examples)) * me)

        gain[a] = total_me - cur_gain

    return max(gain, key=gain.get)


# Filters a dataset based on the filter_item value in the filter_column
def filter_list(items, filter_item, filter_column):
    filtered_items = []

    for i in items:
        if i[filter_column] == filter_item:
            filtered_items.append(i)

    return filtered_items


# Removes the given attribute from a dataset.
def remove_attribute(items, attribute):
    new_items = []
    for i in items:
        new_dict = dict(i)
        del new_dict[attribute]
        new_items.append(new_dict)

    return new_items


# Gets the most common label in items.
def get_common_labels(items):
    labels = {}

    for i in items:
        if i['label'] in labels:
            labels[i['label']] += 1
        else:
            labels[i['label']] = 1

    return max(labels, key=labels.get)


# Prints the result for a given data entry.
def get_result(root_node, input, labels):
    if str(root_node) in labels:
        return str(root_node)

    new_node = root_node.branches[input[str(root_node)]]
    return get_result(new_node, input, labels)


# Converts all attributes in the given numerical column to 'high' or 'low' categories based on the median of the set.
def convert_to_categorical_median(examples, convert_column):
    vals = []

    for i in examples:
        vals.append(i[convert_column])

    vals.sort()
    median = vals[int(len(vals) / 2)]

    for i in examples:
        if i[convert_column] > median:
            i[convert_column] = 'high'
        else:
            i[convert_column] = 'low'


# Counts up the number of times each value of each attribute is used for the given label. This can be used to fill in
# blank labels in a dataset. Blank values should be denoted by unknown.
def get_counts_for_label(examples, target_label):
    counts = {}

    for i in examples[0]:
        if i != 'label' and i != 'value':
            counts[i] = {}

    for i in examples:
        for j in i:
            if i['label'] == target_label:
                if j != 'label' and j != 'value' and i[j] != 'unknown':
                    if not i[j] in counts[j]:
                        counts[j][i[j]] = 1
                    else:
                        counts[j][i[j]] += 1

    return counts

# This function creates a tree based on the cars training data. To do this it reads in the data from the training
# data and converts any numerical category to numerical by using the median of the dataset. It does the same for the test
# data and then creates a series of decision trees based off the training data using heights 1-6 and using 3 different
# methods to generate the purity of the set and prints out the training and test error for each tree.
def create_cars_tree():
    labels = ['unacc', 'acc', 'good', 'vgood']
    attributes = {'buying': ['vhigh', 'high', 'med', 'low'],
                  'maint': ['vhigh', 'high', 'med', 'low'],
                  'doors': ['2', '3', '4', '5more'],
                  'persons': ['2', '4', 'more'],
                  'lug_boot': ['small', 'med', 'big'],
                  'safety': ['low', 'med', 'high']}

    attribute_headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

    file = open('cars_data/train.csv', 'r')

    examples = []
    for l in file:
        temp = {}
        split_line = l.split(',')
        for i in range(len(split_line)):
            temp[attribute_headers[i]] = split_line[i].strip()
        temp['value'] = 1
        examples.append(temp)

    file = open('cars_data/test.csv', 'r')

    test = []
    for l in file:
        temp = {}
        split_line = l.split(',')
        for i in range(len(split_line)):
            temp[attribute_headers[i]] = split_line[i].strip()
        temp['value'] = 1
        test.append(temp)

    for x in range(1,7):
        for test_mode in ['entropy', 'gini', 'me']:
            root_node = id3(examples, attributes, labels, test_mode, 0, x)
            train_error_count = 0
            test_error_count = 0
            for i in examples:
                if get_result(root_node, i, labels) != i['label']:
                    train_error_count += 1
            for i in test:
                if get_result(root_node, i, labels) != i['label']:
                    test_error_count += 1

            print('Tree length: {} Tree Mode: {}\tTest Error: {} Train Error: {}'
                  .format(x, test_mode[:4], train_error_count / len(examples), test_error_count / len(examples)))


# This function creates a tree based on the bank training data. To do this it reads in the data from the training data
# and converts any numerical category to numerical by using the median of the dataset. It does the same for the test
# data and then creates a series of decision trees based off the training data using heights 1-16 and using 3 different
# methods to generate the purity of the set and prints out the training and test error for each tree.
def create_bank_tree():
    labels = ['yes', 'no']
    attributes = {'age': ['low', 'high'],
                  'job': ['admin.','unknown','unemployed','management','housemaid','entrepreneur','student',
                          'blue-collar','self-employed','retired','technician','services'],
                  'marital': ['married','divorced','single'],
                  'education': ['unknown','secondary','primary','tertiary'],
                  'default': ['yes','no'],
                  'balance': ['low', 'high'],
                  'housing': ['yes','no'],
                  'loan': ['yes','no'],
                  'contact': ['unknown','telephone','cellular'],
                  'day': ['low', 'high'],
                  'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                  'duration': ['low', 'high'],
                  'campaign': ['low', 'high'],
                  'pdays': ['low', 'high'],
                  'previous': ['low', 'high'],
                  'poutcome': ['unknown','other','failure','success'],
                  }

    attribute_headers = list(attributes.keys())
    attribute_headers.append('label')

    file = open('bank_data/train.csv', 'r')

    examples = []
    for l in file:
        temp = {}
        split_line = l.split(',')
        for i in range(len(split_line)):
            temp[attribute_headers[i]] = split_line[i].strip()
        temp['value'] = 1
        examples.append(temp)

    convert_to_categorical_median(examples, 'age')
    convert_to_categorical_median(examples, 'balance')
    convert_to_categorical_median(examples, 'day')
    convert_to_categorical_median(examples, 'duration')
    convert_to_categorical_median(examples, 'campaign')
    convert_to_categorical_median(examples, 'pdays')
    convert_to_categorical_median(examples, 'previous')

    file = open('bank_data/test.csv', 'r')

    test_data = []
    for l in file:
        temp = {}
        split_line = l.split(',')
        for i in range(len(split_line)):
            temp[attribute_headers[i]] = split_line[i].strip()
        temp['value'] = 1
        test_data.append(temp)

    convert_to_categorical_median(test_data, 'age')
    convert_to_categorical_median(test_data, 'balance')
    convert_to_categorical_median(test_data, 'day')
    convert_to_categorical_median(test_data, 'duration')
    convert_to_categorical_median(test_data, 'campaign')
    convert_to_categorical_median(test_data, 'pdays')
    convert_to_categorical_median(test_data, 'previous')

    for x in range(1, 17):
        for test_mode in ['entropy', 'gini', 'me']:
            root_node = id3(examples, attributes, labels, test_mode, 0, x)
            train_error_count = 0
            test_error_count = 0
            for i in examples:
                if get_result(root_node, i, labels) != i['label']:
                    train_error_count += 1
            for i in test_data:
                if get_result(root_node, i, labels) != i['label']:
                    test_error_count += 1

            print('Tree length: {} Tree Mode: {}\tTrain Error: {} Test Error: {}'
                  .format(x, test_mode[:4], train_error_count / len(examples), test_error_count / len(examples)))

    common_vals = {}
    common_vals['yes'] = get_counts_for_label(examples, 'yes')
    common_vals['no'] = get_counts_for_label(examples, 'no')

    for i in examples:
        for j in i:
            if i[j] == 'unknown':
                i[j] = max(common_vals[i['label']][j], key=common_vals[i['label']][j].get)

    for x in range(1, 17):
        for test_mode in ['entropy', 'gini', 'me']:
            root_node = id3(examples, attributes, labels, test_mode, 0, x)
            train_error_count = 0
            test_error_count = 0
            for i in examples:
                if get_result(root_node, i, labels) != i['label']:
                    train_error_count += 1
            for i in test_data:
                if get_result(root_node, i, labels) != i['label']:
                    test_error_count += 1

            print('Tree length: {} Tree Mode: {}\tTrain Error: {} Test Error: {}\\\\'\
                  .format(x, test_mode[:4], train_error_count / len(examples), test_error_count / len(examples)))


if __name__ == '__main__':
    print("Creating Cars Tree\n")
    create_cars_tree()
    print("Creating Bank Tree\n")
    create_bank_tree()


