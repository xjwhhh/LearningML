# encoding=utf-8
import cv2
import time
import logging
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 参考了别人的实现
# 问题1：可能是因为没有进行二值化？算出来的正确率很低，只有10%？还不如瞎猜！
# 直接使用别人的代码（将二值化注释）得到的正确率也很低，但看博文达到了89%，不知其解
# 问题2：我觉得参考的代码在实现经验条件熵的时候是有问题的，改成了自己实现的
# 问题3：在tree.predict中可能出现keyError的问题，不得已进行了键是否存在的检测，并随机返回值。我没有找到原因
# 问题4：但就我所看，问题3的情况不多，对最后结果产生的影响应该不大，但正确率还是很低
# 问题5：使用原来的train_binary数据集计算很慢，用了小一点的数据集
# 虽然有这么多问题，但关键步骤的代码，如经验熵和经验条件熵的计算，ID3算法的各个步骤应该都是正确的

# 好吧发现不是二值化的问题！博主原来的代码我运行了也是很低的正确率，不得其解

total_class = 10


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
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY, cv_img)
    return cv_img


@log
def binaryzation_features(trainset):
    features = []

    for img in trainset:
        img = np.reshape(img, (10, 10))
        cv_img = img.astype(np.uint8)

        img_b = binaryzation(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(img_b)

    features = np.array(features)
    features = np.reshape(features, (-1, 100))

    return features


class Tree(object):
    def __init__(self, node_type, Class=None, feature=None):
        self.node_type = node_type
        self.dict = {}
        self.Class = Class
        self.feature = feature

    def add_tree(self, val, tree):
        self.dict[val] = tree

    def predict(self, features):
        if self.node_type == 'leaf':
            return self.Class
        if (features[self.feature] in self.dict.keys()):
            tree = self.dict[features[self.feature]]
        else:
            if (self.Class is None):
                return random.randint(0, 1)
            else:
                return self.Class
        return tree.predict(features)


def calc_ent(x):
    """
        calculate empirical entropy of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent


def calc_condition_ent(train_feature, train_label):
    """
        calculate empirical entropy H(y|x)
    """

    # calc ent(y|x)

    ent = 0
    train_feature_set = set(train_feature)
    # print("train_feature_set", train_feature_set)
    for train_feature_value in train_feature_set:
        Di = train_feature[train_feature == train_feature_value]
        label_i = train_label[train_feature == train_feature_value]
        # print("Di", Di)
        train_label_set = set(train_label)
        temp = 0
        # print("train_label_set", train_label_set)
        for train_label_value in train_label_set:
            Dik = Di[label_i == train_label_value]
            # print(Dik)
            if (len(Dik) != 0):
                p = float(len(Dik)) / len(Di)
                logp = np.log2(p)
                temp -= p * logp
        ent += float(len(Di)) / len(train_feature) * temp
    return ent


def recurse_train(train_set, train_label, features, epsilon):
    global total_class

    LEAF = 'leaf'
    INTERNAL = 'internal'

    # 步骤1——如果train_set中的所有实例都属于同一类Ck
    label_set = set(train_label)
    # print(label_set)
    if len(label_set) == 1:
        return Tree(LEAF, Class=label_set.pop())

    # 步骤2——如果features为空

    class_count0 = 0
    class_count1 = 0

    for i in range(len(train_label)):
        if (train_label[i] == 1):
            class_count1 += 1
        else:
            class_count0 += 1

    if (class_count0 >= class_count1):
        max_class = 0
    else:
        max_class = 0

    if features is None:
        return Tree(LEAF, Class=max_class)

    if len(features) == 0:
        return Tree(LEAF, Class=max_class)

    # 步骤3——计算信息增益
    max_feature = 0
    max_gda = 0

    D = train_label
    HD = calc_ent(D)
    for feature in features:
        A = np.array(train_set[:, feature].flat)
        gda = HD - calc_condition_ent(A, D)

        if gda > max_gda:
            max_gda, max_feature = gda, feature

    # 步骤4——小于阈值
    if max_gda < epsilon:
        return Tree(LEAF, Class=max_class)

    # 步骤5——构建非空子集
    sub_features = features.remove(max_feature)
    tree = Tree(INTERNAL, feature=max_feature)

    feature_col = np.array(train_set[:, max_feature].flat)
    feature_value_list = set([feature_col[i] for i in range(feature_col.shape[0])])
    for feature_value in feature_value_list:

        index = []
        for i in range(len(train_label)):
            if train_set[i][max_feature] == feature_value:
                index.append(i)

        sub_train_set = train_set[index]
        sub_train_label = train_label[index]

        sub_tree = recurse_train(sub_train_set, sub_train_label, sub_features, epsilon)
        tree.add_tree(feature_value, sub_tree)

    return tree


@log
def train(train_set, train_label, features, epsilon):
    # print(features)
    return recurse_train(train_set, train_label, features, epsilon)


@log
def predict(test_set, tree):
    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    raw_data = pd.read_csv('../data/train_binary2.csv', header=0)
    data = raw_data.values

    images = data[0:, 1:]
    labels = data[:, 0]

    # 图片二值化
    features = binaryzation_features(images)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
                                                                                random_state=1)

    # print(train_features.shape)
    tree = train(train_features, train_labels, [i for i in range(99)], 0.1)
    test_predict = predict(test_features, tree)
    # print(test_predict)
    score = accuracy_score(test_labels, test_predict)

    print("The accuracy score is ", score)
