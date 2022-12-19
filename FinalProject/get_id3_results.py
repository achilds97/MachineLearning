import numpy as np

import DecisionTree.id3 as id3
import EnsembleLearning.boosting as boosting
import sys

sys.path.insert(0, 'C:/Users/Alex Childs/OneDrive/Documents/2022 Fall Semester/CS5350/MachineLearning/DecisionTree')

def id3_with_only_categorical_columns():
    labels=['0','1']
    attributes={'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
                              'Without-pay', 'Never-worked', '?'],
                'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc',
                              '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th',
                              'Preschool', '?'],
                'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                                   'Married-spouse-absent', 'Married-AF-spouse', '?'],
                'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                               'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                               'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                               'Armed-Forces', '?'],
                'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?'],
                'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?'],
                'sex': ['Male', 'Female', '?'],
                'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                                   'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba',
                                   'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico',
                                   'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
                                   'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand',
                                   'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands',
                                   '?'],
                'age': ['low', 'high'],
                'fnlwgt': ['low', 'high'],
                'education-num': ['low', 'high'],
                'capital-gain': ['low', 'high'],
                'capital-loss': ['low', 'high'],
                'hours-per-week': ['low', 'high'],
                }

    attribute_headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                         'native-country', 'label']

    data_columns = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    test_data_columns = [2, 4, 6, 7, 8, 9, 10, 14, 15]

    file = open('income2022f/train_final.csv')

    examples = []
    for l in file:
        temp = {}
        split_line = l.split(',')
        column_count = 0
        for i in range(len(split_line)):
            temp[attribute_headers[column_count]] = split_line[i].strip()
            column_count += 1
        temp['value'] = 1
        examples.append(temp)

    id3.convert_to_categorical_median(examples, 'age')
    id3.convert_to_categorical_median(examples, 'fnlwgt')
    id3.convert_to_categorical_median(examples, 'education-num')
    id3.convert_to_categorical_median(examples, 'capital-gain')
    id3.convert_to_categorical_median(examples, 'capital-loss')
    id3.convert_to_categorical_median(examples, 'hours-per-week')

    file = open('income2022f/test_final.csv')

    test = []
    for l in file:
        temp = {}
        split_line = l.split(',')
        column_count = 0
        for i in range(1, len(split_line)):
            temp[attribute_headers[column_count]] = split_line[i].strip()
            column_count += 1
        temp['value'] = 1
        test.append(temp)

    id3.convert_to_categorical_median(test, 'age')
    id3.convert_to_categorical_median(test, 'fnlwgt')
    id3.convert_to_categorical_median(test, 'education-num')
    id3.convert_to_categorical_median(test, 'capital-gain')
    id3.convert_to_categorical_median(test, 'capital-loss')
    id3.convert_to_categorical_median(test, 'hours-per-week')

    # for x in range(2, 20):
    #     root_node = id3.id3(examples, attributes, labels, 'gini', 0, x)
    #     #test_root_node = id3.id3(test, attributes, labels, 'entropy', 0, 10)
    #     train_error_count = 0
    #     test_error_count = 0
    #     for i in examples:
    #         if id3.get_result(root_node, i, labels) != i['label']:
    #             train_error_count += 1
    #     print(train_error_count / len(examples))

    weights = np.array([1 / len(examples) for i in range(len(examples))])
    boost_set = []
    alpha_set = []

    training_errors = []
    test_errors = []

    for i in range(10):
        for j in range(len(examples)):
            examples[j]['value'] = weights[j]

        root_node = id3.id3(examples, attributes, labels, 'entropy', 0, 1)
        boost_set.append(root_node)

        weights, alpha = boosting.update_weights(root_node, weights, examples)
        alpha_set.append(alpha)

        all_results = []
        missed_data = 0
        for j in examples:
            boost_result = boosting.get_boost_result(boost_set, j, alpha_set)
            all_results.append([boost_result, j['label']])
            if boost_result != j['label']:
                missed_data += 1

        training_errors.append(missed_data / len(all_results))

    for i in boost_set:
        print(str(i))

    f = open('result.csv', 'w')
    f.write('ID,Prediction\n')

    cur_count = 0

    for i in test:

        cur_count += 1
        result = boosting.get_boost_result(boost_set, i, alpha_set)

        f.write(str(cur_count) + ',' + str(result) + '\n')

    # cur_count = 1

    # for i in test:
    #     f.write(str(cur_count) + ',' + str(id3.get_result(root_node, i, labels)) + '\n')
    #     cur_count += 1

    f.close()




if __name__ == '__main__':
    id3_with_only_categorical_columns()
