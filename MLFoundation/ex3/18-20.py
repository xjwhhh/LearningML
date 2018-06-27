import numpy as np


def data_load(file_path):
    # open file and read lines
    f = open(file_path)
    try:
        lines = f.readlines()
    finally:
        f.close()

    # create features and labels array
    example_num = len(lines)
    feature_dimension = len(lines[0].strip().split())

    features = np.zeros((example_num, feature_dimension))
    features[:, 0] = 1
    labels = np.zeros((example_num, 1))

    for index, line in enumerate(lines):
        # items[0:-1]--features   items[-1]--label
        items = line.strip().split(' ')
        # get features
        features[index, 1:] = [float(str_num) for str_num in items[0:-1]]

        # get label
        labels[index] = float(items[-1])

    return features, labels


# gradient descent
def gradient_descent(X, y, w):
    # -YnWtXn
    tmp = -y * (np.dot(X, w))

    # θ(-YnWtXn) = exp(tmp)/1+exp(tmp)
    # weight_matrix = np.array([math.exp(_)/(1+math.exp(_)) for _ in tmp]).reshape(len(X), 1)
    weight_matrix = np.exp(tmp) / ((1 + np.exp(tmp)) * 1.0)
    gradient = 1 / (len(X) * 1.0) * (sum(weight_matrix * -y * X).reshape(len(w), 1))

    return gradient


# gradient descent
def stochastic_gradient_descent(X, y, w):
    # -YnWtXn
    tmp = -y * (np.dot(X, w))

    # θ(-YnWtXn) = exp(tmp)/1+exp(tmp)
    # weight = math.exp(tmp[0])/((1+math.exp(tmp[0]))*1.0)
    weight = np.exp(tmp) / ((1 + np.exp(tmp)) * 1.0)

    gradient = weight * -y * X
    return gradient.reshape(len(gradient), 1)


# LinearRegression Class
class LinearRegression:

    def __init__(self):
        pass

    # fit model
    def fit(self, X, y, Eta=0.001, max_iteration=2000, sgd=False):
        # ∂E/∂w = 1/N * ∑θ(-YnWtXn)(-YnXn)
        self.__w = np.zeros((len(X[0]), 1))

        # whether use stochastic gradient descent
        if not sgd:
            for i in range(max_iteration):
                self.__w = self.__w - Eta * gradient_descent(X, y, self.__w)
        else:
            index = 0
            for i in range(max_iteration):
                if (index >= len(X)):
                    index = 0
                self.__w = self.__w - Eta * stochastic_gradient_descent(np.array(X[index]), y[index], self.__w)
                index += 1

    # predict
    def predict(self, X):
        binary_result = np.dot(X, self.__w) >= 0
        return np.array([(1 if _ > 0 else -1) for _ in binary_result]).reshape(len(X), 1)

    # get vector w
    def get_w(self):
        return self.__w

    # score(error rate)
    def score(self, X, y):
        predict_y = self.predict(X)
        return sum(predict_y != y) / (len(y) * 1.0)


if __name__ == '__main__':
    # 18
    # training model
    (X, Y) = data_load("hw3_train.dat")
    lr = LinearRegression()
    lr.fit(X, Y, max_iteration=2000)

    # get 0/1 error in test data
    test_X, test_Y = data_load("hw3_test.dat")
    print("E_out: ", lr.score(test_X, test_Y))

    # 19
    # training model
    (X, Y) = data_load("hw3_train.dat")
    lr_eta = LinearRegression()
    lr_eta.fit(X, Y, 0.01, 2000)

    # get 0/1 error in test data
    test_X, test_Y = data_load("hw3_test.dat")
    print("E_out: ", lr_eta.score(test_X, test_Y))

    # 20
    (X, Y) = data_load("hw3_train.dat")
    lr_sgd = LinearRegression()
    lr_sgd.fit(X, Y, sgd=True, max_iteration=2000)

    # get 0/1 error in test data
    test_X, test_Y = data_load("hw3_test.dat")
    print("E_out: ", lr_sgd.score(test_X, test_Y))
