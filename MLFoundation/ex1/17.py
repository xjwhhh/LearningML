import numpy
import random


class RandomPLA(object):
    def __init__(self, dimension, count):
        self.__dimension = dimension
        self.__count = count

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
        x_train = numpy.zeros((self.__count, self.__dimension))
        y_train = numpy.zeros((self.__count, 1))
        random_list = self.random_matrix(path)
        for i in range(self.__count):
            for j in range(self.__dimension):
                x_train[i, j] = random_list[i][j]
            y_train[i, 0] = random_list[i][self.__dimension]
        return x_train, y_train

    def iteration_count(self, path):
        count = 0
        x_train, y_train = self.train_matrix(path)
        w = numpy.zeros((self.__dimension, 1))
        while True:
            flag = 0
            for i in range(self.__count):
                if numpy.dot(x_train[i, :], w)[0] * y_train[i, 0] <= 0:
                    w += 0.5 * y_train[i, 0] * x_train[i, :].reshape(5, 1)
                    count += 1
                    flag = 1
            if flag == 0:
                break
        return count


sum = 0
for i in range(2000):
    perceptron = RandomPLA(5, 400)
    sum += perceptron.iteration_count('hw1_15_train.dat')
print(sum / 2000.0)
