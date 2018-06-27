import numpy as np


def read_input_data(path):
    x = []
    y = []
    for line in open(path).readlines():
        items = line.strip().split(' ')
        tmp_x = []
        for i in range(0, len(items) - 1): tmp_x.append(float(items[i]))
        x.append(tmp_x)
        y.append(float(items[-1]))
    return np.array(x), np.array(y)


def calculate_Ein(x, y):
    # calculate median of interval & negative infinite & positive infinite
    thetas = np.array([float("-inf")] + [(x[i] + x[i + 1]) / 2 for i in range(0, x.shape[0] - 1)] + [float("inf")])
    Ein = x.shape[0]
    sign = 1
    target_theta = 0.0
    # positive and negative rays
    for theta in thetas:
        y_positive = np.where(x > theta, 1, -1)
        y_negative = np.where(x < theta, 1, -1)
        error_positive = sum(y_positive != y)
        error_negative = sum(y_negative != y)
        if error_positive > error_negative:
            if Ein > error_negative:
                Ein = error_negative
                sign = -1
                target_theta = theta
        else:
            if Ein > error_positive:
                Ein = error_positive
                sign = 1
                target_theta = theta
    return Ein, target_theta, sign


if __name__ == '__main__':
    # 19
    x, y = read_input_data("hw2_train.dat")
    # record optimal descision stump parameters
    Ein = x.shape[0]
    theta = 0
    sign = 1
    index = 0
    # multi decision stump optimal process
    for i in range(0, x.shape[1]):
        input_x = x[:, i]
        input_data = np.transpose(np.array([input_x, y]))
        input_data = input_data[np.argsort(input_data[:, 0])]
        curr_Ein, curr_theta, curr_sign = calculate_Ein(input_data[:, 0], input_data[:, 1])
        if Ein > curr_Ein:
            Ein = curr_Ein
            theta = curr_theta
            sign = curr_sign
            index = i
    print((Ein * 1.0) / x.shape[0])
    # 20
    # test process
    test_x, test_y = read_input_data("hw2_test.dat")
    test_x = test_x[:, index]
    predict_y = np.array([])
    if sign == 1:
        predict_y = np.where(test_x > theta, 1.0, -1.0)
    else:
        predict_y = np.where(test_x < theta, 1.0, -1.0)
    Eout = sum(predict_y != test_y)
    print((Eout * 1.0) / test_x.shape[0])
