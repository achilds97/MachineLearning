import math


def forward_propagate():
    weights = [[[None, -1, 1], [None, -2, 2], [None, -3, 3]],
               [[None, -1, 1], [None, -2, 2], [None, -3, 3]],
               [[None, -1], [None, 2], [None, -1.5]]]

    widths = [3, 3, 3]
    out = [[[] for j in range(i)] for i in widths]
    inputs = [1, 1, 1]

    for i in range(len(out)):
        for j in range(widths[i]):
            if i == 0:
                out[i][j] = inputs[j]
            elif j == 0:
                out[i][j] = 1
            else:
                cur_vals = []
                for k in range(widths[i]):
                    cur_vals.append(out[i-1][k] * weights[i-1][k][j])

                    out[i][j] = sigmoid(cur_vals)

    y = sum([out[-1][i] * weights[-1][i][1] for i in range(widths[-1])])

    print(y)
    print(out)


def sigmoid(s):
    return 1 / (1 + math.exp(sum(s) * -1))


def loss(expected, actual):
    return 0.5 * (actual - expected)**2


if __name__ == '__main__':
    forward_propagate()