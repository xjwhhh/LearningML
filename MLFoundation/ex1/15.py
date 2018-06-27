import numpy

class NaiveCyclePLA(object):
    def __init__(self, dimension, count):
        self.__dimension = dimension
        self.__count = count

    # get data
    def train_matrix(self, path):
        training_set = open(path)
        x_train = numpy.zeros((self.__count, self.__dimension))
        y_train = numpy.zeros((self.__count, 1))
        x = []
        x_count = 0
        for line in training_set:
            # add 1 dimension manually
            x.append(1)
            for str in line.split(' '):
                if len(str.split('\t')) == 1:
                    x.append(float(str))
                else:
                    x.append(float(str.split('\t')[0]))
                    y_train[x_count, 0] = int(str.split('\t')[1].strip())
            x_train[x_count, :] = x
            x = []
            x_count += 1
        return x_train, y_train

    def iteration_count(self, path):
        count = 0
        x_train, y_train = self.train_matrix(path)
        w = numpy.zeros((self.__dimension, 1))
        # loop until all x are classified right
        while True:
            flag = 0
            for i in range(self.__count):
                if numpy.dot(x_train[i, :], w)[0] * y_train[i, 0] <= 0:
                    w += y_train[i, :] * x_train[i, :].reshape(5, 1)
                    count += 1
                    flag = 1
            if flag == 0:
                break
        return count


if __name__ == '__main__':
    perceptron = NaiveCyclePLA(5, 400)
    print(perceptron.iteration_count("hw1_15_train.dat"))
