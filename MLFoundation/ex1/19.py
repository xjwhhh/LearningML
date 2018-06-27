import numpy
import random
import copy


class Pocket(object):
    def __init__(self, dimension, train_count, test_count):
        self.__dimension = dimension
        self.__train_count = train_count
        self.__test_count = test_count

    def random_matrix(self, path):
        training_set = open(path)
        random_list = []
        x = []
        x_count = 0
        for line in training_set:
            x.append(1)
            for str in line.split(' '):
                if len(str.split('\t')) == 1:
                    x.append(float(str))
                else:
                    x.append(float(str.split('\t')[0]))
                    x.append(int(str.split('\t')[1].strip()))
            random_list.append(x)
            x = []
            x_count += 1
        random.shuffle(random_list)
        return random_list

    def train_matrix(self, path):
        x_train = numpy.zeros((self.__train_count, self.__dimension))
        y_train = numpy.zeros((self.__train_count, 1))
        random_list = self.random_matrix(path)
        for i in range(self.__train_count):
            for j in range(self.__dimension):
                x_train[i, j] = random_list[i][j]
            y_train[i, 0] = random_list[i][self.__dimension]
        return x_train, y_train

    def iteration(self, path):
        count = 0
        x_train, y_train = self.train_matrix(path)
        w = numpy.zeros((self.__dimension, 1))
        for i in range(self.__train_count):
            if numpy.dot(x_train[i, :], w)[0] * y_train[i, 0] <= 0:
                w += 0.5 * y_train[i, 0] * x_train[i, :].reshape(5, 1)
                count += 1
            if count == 50:
                break
        return w

    def test_matrix(self, test_path):
        x_test = numpy.zeros((self.__test_count, self.__dimension))
        y_test = numpy.zeros((self.__test_count, 1))
        test_set = open(test_path)
        x = []
        x_count = 0
        for line in test_set:
            x.append(1)
            for str in line.split(' '):
                if len(str.split('\t')) == 1:
                    x.append(float(str))
                else:
                    x.append(float(str.split('\t')[0]))
                    y_test[x_count, 0] = (int(str.split('\t')[1].strip()))
            x_test[x_count, :] = x
            x = []
            x_count += 1
        return x_test, y_test

    # 验证
    def test_error(self, train_path, test_path):
        w = self.iteration(train_path)
        x_test, y_test = self.test_matrix(test_path)
        count = 0.0
        for i in range(self.__test_count):
            if numpy.dot(x_test[i, :], w)[0] * y_test[i, 0] <= 0:
                count += 1
        return count / self.__test_count


if __name__ == '__main__':
    average_error_rate = 0
    for i in range(2000):
        my_Pocket = Pocket(5, 500, 500)
        average_error_rate += my_Pocket.test_error('hw1_18_train.dat', 'hw1_18_test.dat')
    print(average_error_rate / 2000.0)
