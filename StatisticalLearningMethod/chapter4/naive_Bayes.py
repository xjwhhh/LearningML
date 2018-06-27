import cv2
import time
import logging
import numpy as np
import pandas as pd

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


# 二值化,将图片进行二值化的目的是确定每个特征可选的值只有两种，对应于train方法里conditional_probability最后一个维度的长度2
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img


@log
def train(train_set, train_labels):
    class_num = len(set(train_labels))
    feature_num = len(train_set[0])
    prior_probability = np.zeros(class_num)  # 先验概率
    conditional_probability = np.zeros((class_num, feature_num, 2))  # 条件概率
    print(conditional_probability.shape)

    for i in range(len(train_labels)):
        img = binaryzation(train_set[i])  # 图片二值化
        label = train_labels[i]

        prior_probability[label] += 1

        for j in range(feature_num):
            conditional_probability[label][j][img[j]] += 1

    # 贝叶斯估计，因为分母都相同，所以先验概率和条件概率都不用除以分母
    prior_probability += 1
    for label in set(train_labels):
        for j in range(feature_num):
            conditional_probability[label][j][0] += 1
            conditional_probability[label][j][0] /= (len(train_labels[train_labels == label]) + 2 * 1)
            conditional_probability[label][j][1] += 1
            conditional_probability[label][j][1] /= (len(train_labels[train_labels == label]) + 2 * 1)

    # print(prior_probability)
    # print(conditional_probability)
    return prior_probability, conditional_probability


@log
def predict(test_features, prior_probability, conditional_probability):
    result = []
    for test in test_features:
        img = binaryzation(test)

        max_label = 0
        max_probability = 0

        for i in range(len(prior_probability)):

            # print("label",i)
            probability = prior_probability[i]
            for j in range(len(img)):  # 特征长度
                # print("j",j)
                probability *= int(conditional_probability[i][j][img[j]])
            if max_probability < probability:
                max_probability = probability
                max_label = i
        result.append(max_label)
    return np.array(result)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values

    imgs = data[0:2000, 1:]
    labels = data[0:2000, 0]

    # print(imgs.shape)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33,
                                                                                random_state=1)

    prior_probability, conditional_probability = train(train_features, train_labels)
    test_predict = predict(test_features, prior_probability, conditional_probability)
    score = accuracy_score(test_labels, test_predict)
    print("The accuracy score is ", score)
