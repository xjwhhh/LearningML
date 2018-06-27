import cv2
import time
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

total_class = 10


# 这里选用了一个比较小的数据集，因为过大的数据集会导致栈溢出


def log(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.debug('start %s()' % func.__name__)
        ret = func(*args, **kwargs)

        end_time = time.time()
        logging.debug('end %s(), cost %s seconds' % (func.__name__, end_time - start_time))

        return ret

    return wrapper


# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img


@log
def binaryzation_features(trainset):
    features = []

    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)

        img_b = binaryzation(cv_img)
        features.append(img_b)

    features = np.array(features)
    features = np.reshape(features, (-1, 784))

    return features


class TreeNode(object):
    """决策树节点"""

    def __init__(self, **kwargs):
        '''
        attr_index: 属性编号
        attr: 属性值
        label: 类别（y）
        left_chuld: 左子结点
        right_child: 右子节点
        '''
        self.attr_index = kwargs.get('attr_index')
        self.attr = kwargs.get('attr')
        self.label = kwargs.get('label')
        self.left_child = kwargs.get('left_child')
        self.right_child = kwargs.get('right_child')


# 计算数据集的基尼指数
def gini_train_set(train_label):
    train_label_value = set(train_label)
    gini = 0.0
    for i in train_label_value:
        train_label_temp = train_label[train_label == i]
        pk = float(len(train_label_temp)) / len(train_label)
        gini += pk * (1 - pk)
    return gini


# 计算一个特征不同切分点的基尼指数，并返回最小的
def gini_feature(train_feature, train_label):
    train_feature_value = set(train_feature)
    min_gini = float('inf')
    return_feature_value = 0
    for i in train_feature_value:
        train_feature_class1 = train_feature[train_feature == i]
        label_class1 = train_label[train_feature == i]
        # train_feature_class2 = train_feature[train_feature != i]
        label_class2 = train_label[train_feature != i]
        D1 = float(len(train_feature_class1)) / len(train_feature)
        D2 = 1 - D1
        if (len(label_class1) == 0):
            p1 = 0
        else:
            p1 = float(len(label_class1[label_class1 == label_class1[0]])) / len(label_class1)
        if (len(label_class2) == 0):
            p2 = 0
        else:
            p2 = float(len(label_class2[label_class2 == label_class2[0]])) / len(label_class2)
        gini = D1 * 2 * p1 * (1 - p1) + D2 * 2 * p2 * (1 - p2)
        if min_gini > gini:
            min_gini = gini
            return_feature_value = i
    return min_gini, return_feature_value


def get_best_index(train_set, train_label, feature_indexes):
    '''
    :param train_set: 给定数据集
    :param train_label: 数据集对应的标记
    :return: 最佳切分点，最佳切分变量
    求给定切分点集合中的最佳切分点和其对应的最佳切分变量
    '''
    min_gini = float('inf')
    feature_index = 0
    return_feature_value = 0
    for i in range(len(train_set[0])):
        if i in feature_indexes:
            train_feature = train_set[:, i]
            gini, feature_value = gini_feature(train_feature, train_label)
            if gini < min_gini:
                min_gini = gini
                feature_index = i
                return_feature_value = feature_value
    return feature_index, return_feature_value


# 根据最有特征和最优切分点划分数据集
def divide_train_set(train_set, train_label, feature_index, feature_value):
    left = []
    right = []
    left_label = []
    right_label = []
    for i in range(len(train_set)):
        line = train_set[i]
        if line[feature_index] == feature_value:
            left.append(line)
            left_label.append(train_label[i])
        else:
            right.append(line)
            right_label.append(train_label[i])
    return np.array(left), np.array(right), np.array(left_label), np.array(right_label)


@log
def build_tree(train_set, train_label, feature_indexes):
    # 查看是否满足停止条件
    train_label_value = set(train_label)
    if len(train_label_value) == 1:
        print("a")
        return TreeNode(label=train_label[0])

    if feature_indexes is None:
        print("b")
        return TreeNode(label=train_label[0])

    if len(feature_indexes) == 0:
        print("c")
        return TreeNode(label=train_label[0])

    feature_index, feature_value = get_best_index(train_set, train_label, feature_indexes)
    # print("feature_index",feature_index)

    left, right, left_label, right_label = divide_train_set(train_set, train_label, feature_index, feature_value)

    feature_indexes.remove(feature_index)
    # print("feature_indexes",feature_indexes)

    left_branch = build_tree(left, left_label, feature_indexes)
    right_branch = build_tree(right, right_label, feature_indexes)
    return TreeNode(left_child=left_branch,
                    right_child=right_branch,
                    attr_index=feature_index,
                    attr=feature_value)

# @log
# def prune(tree):


def predict_one(node, test):
    while node.label is None:
        if test[node.attr_index] == node.attr:
            node = node.left_child
        else:
            node = node.right_child
    return node.label


@log
def predict(tree, test_set):
    result = []
    for test in test_set:
        label = predict_one(tree, test)
        result.append(label)
    return result


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    raw_data = pd.read_csv('../data/train_binary1.csv', header=0)
    data = raw_data.values

    imgs = data[0:, 1:]
    labels = data[:, 0]

    print(imgs.shape)

    # 图片二值化
    # features = binaryzation_features(imgs)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33,
                                                                                random_state=23323)

    print(type(train_features))
    tree = build_tree(train_features, train_labels, [i for i in range(784)])
    test_predict = predict(tree, test_features)
    score = accuracy_score(test_labels, test_predict)

    print("The accuracy score is ", score)
