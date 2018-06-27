import pandas as pd
import numpy as np
import cv2
import logging
import time

from math import sqrt
from collections import namedtuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def log(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.debug('start %s()' % func.__name__)
        ret = func(*args, **kwargs)

        end_time = time.time()
        logging.debug('end %s(), cost %s seconds' % (func.__name__, end_time - start_time))

        return ret

    return wrapper


def get_hog_features(trainset):
    # 利用opencv获取图像hog特征

    features = []

    hog = cv2.HOGDescriptor('../hog.xml')

    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features, (-1, 324))

    return features


def predict(test_set, kd_tree):
    predict = []

    for i in range(len(test_set)):
        predict.append(find_nearest(kd_tree, test_set[i]).label)

    return np.array(predict)


# 构造kdTree搜索
# 现在的实现是最近邻，
# 问题1：怎么保存每个结点对应的label，现在的实现似乎成功了，但我不确定
# 问题2：速度非常慢

class KdNode(object):
    def __init__(self, dom_elt, split, left, right, label):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split  # 整数（进行分割维度的序号）
        self.left = left  # 该结点分割超平面左子空间构成的kd-tree
        self.right = right  # 该结点分割超平面右子空间构成的kd-tree
        self.label = label


class KdTree(object):

    @log
    def __init__(self, data, labels):
        k = len(data[0])  # 数据维度

        def create_node(split, data_set, labels):  # 按第split维划分数据集,创建KdNode

            # print(len(data_set))
            if (len(data_set) == 0):
                return None

            sort_index = data_set[:, split].argsort()
            data_set = data_set[sort_index]
            labels = labels[sort_index]
            # print(data_set)

            split_pos = len(data_set) // 2
            # print(split_pos)
            median = data_set[split_pos]  # 中位数分割点
            label = labels[split_pos]
            split_next = (split + 1) % k  # cycle coordinates

            # 递归的创建kd树
            return KdNode(median, split,
                          create_node(split_next, data_set[:split_pos], labels[:split_pos]),  # 创建左子树
                          create_node(split_next, data_set[split_pos + 1:], labels[split_pos + 1:]),  # 创建右子树
                          label)

        self.root = create_node(0, data, labels)  # 从第0维分量开始构建kd树,返回根节点


# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple("Result_tuple", "nearest_point  nearest_dist  nodes_visited label")


@log
def find_nearest(tree, point):
    k = len(point)  # 数据维度

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"), 0, 0)  # python中用float("inf")和float("-inf")表示正负无穷

        nodes_visited = 1

        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的“轴”

        if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left  # 下一个访问节点为左子树根节点
            further_node = kd_node.right  # 同时记录下右子树
        else:  # 目标离右子树更近
            nearer_node = kd_node.right  # 下一个访问节点为右子树根节点
            further_node = kd_node.left
        if (nearer_node is None):
            label = 0
        else:
            label = nearer_node.label

        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

        nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist  # 更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist:  # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited, temp1.label)  # 不相交则可以直接返回，不用继续判断

        # ----------------------------------------------------------------------
        # 计算目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist:  # 如果“更近”
            nearest = pivot  # 更新最近点
            dist = temp_dist  # 更新最近距离
            max_dist = dist  # 更新超球体半径
            label = kd_node

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:  # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point  # 更新最近点
            dist = temp2.nearest_dist  # 更新最近距离
            label = temp2.label

        return result(nearest, dist, nodes_visited, label)

    return travel(tree.root, point, float("inf"))  # 从根节点开始递归


k = 10

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values

    images = data[0:, 1:]
    labels = data[:, 0]

    features = get_hog_features(images)
    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
                                                                                random_state=1)

    kd_tree = KdTree(train_features, train_labels)

    test_predict = predict(test_features, kd_tree)

    score = accuracy_score(test_labels, test_predict)
    print("The accuracy score is ", score)
