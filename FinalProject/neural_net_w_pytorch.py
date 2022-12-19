from pandas import read_csv
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

train_data = read_csv('income2022f/train_final.csv', header=None)
test_data = read_csv('income2022f/test_final.csv', header=None)


def get_data_train(filename):
    data = read_csv(filename, header=None)
    oe = OrdinalEncoder()
    data[[1, 3, 5, 6, 7, 8, 9, 13]] = oe.fit_transform(data[[1, 3, 5, 6, 7, 8, 9, 13]])
    dataset = data.values
    x = dataset[:, :-1]
    y = dataset[:, -1]

    new_y = []

    for i in y:
        if i == 0:
            new_y.append(-1)
        else:
            new_y.append(1)

    y = np.array(new_y).reshape((len(y), 1))
    return torch.from_numpy(x), torch.from_numpy(y)


def get_data_test(filename):
    data = read_csv(filename, header=None)
    oe = OrdinalEncoder()
    data[[2, 4, 6, 7, 8, 9, 10, 14]] = oe.fit_transform(data[[1, 4, 6, 7, 8, 9, 10, 14]])
    dataset = data.values
    x = dataset[:, :-1]

    return torch.from_numpy(x)

def main():
    x_train, y_train = get_data_train('income2022f/train_final.csv')

    x_train = x_train.to(torch.float32)
    y_train = y_train.to(torch.float32)

    x_test = get_data_test('income2022f/test_final.csv')
    x_test = x_test.to(torch.float32)

    n_input, n_hidden, n_out, batch_size, learning_rate = 14, 20, 1, 100, 0.01

    model = nn.Sequential(nn.Linear(n_input, n_hidden),
                          nn.Sigmoid(),
                          nn.Linear(n_hidden, n_out),
                          nn.Sigmoid())
    print(model)
    torch.randn
    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    losses = []
    for epoch in range(600):
        pred_y = model(x_train)
        loss = loss_function(pred_y, y_train)
        losses.append(loss.item())

        model.zero_grad()
        loss.backward()

        optimizer.step()

    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f"%(learning_rate))
    plt.show()

    pred_test_y = model(x_train)

    f = open('result_nn.csv', 'w')
    f.write('ID,Prediction\n')
    count = 0
    for t in range(len(pred_test_y)):
        print(pred_test_y[t].item())
        if pred_test_y[t].item() >= 0.5:
            count += 1
            f.write(str(t + 1) + ',1\n')
        else:
            f.write(str(t + 1) + ',0\n')

    f.close()
    print(count)


if __name__ == '__main__':
    main()