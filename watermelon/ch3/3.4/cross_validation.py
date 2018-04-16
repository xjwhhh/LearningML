
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

'''
1-st. iris data set importing and visualization using seaborn
'''

sns.set(style="white", color_codes=True)
iris = sns.load_dataset("iris")
X = iris.values[50:150, 0:4]
y = iris.values[50:150, 4]

# iris.plot(kind="scatter", x="sepal_length", y="sepal_width")
# sns.pairplot(iris,hue='species')
# sns.plt.show()

'''
2-nd logistic regression using sklearn
'''


# log-regression lib model
log_model = LogisticRegression()
m = np.shape(X)[0]

# 10-folds CV  10折交叉验证
y_pred = cross_val_predict(log_model, X, y, cv=10)
print(metrics.accuracy_score(y, y_pred))

# LOOCV 留一法
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
accuracy = 0
for train, test in loo.split(X):
    log_model.fit(X[train], y[train])  # fitting
    y_p = log_model.predict(X[test])
    if y_p == y[test]: accuracy += 1
print(accuracy / np.shape(X)[0])

'''
transfusion-blood data set analysis
'''

dataset_transfusion = np.loadtxt('data/transfusion.data', delimiter=",", skiprows=1)
X2 = dataset_transfusion[:, 0:4]
y2 = dataset_transfusion[:, 4]


# log-regression lib model
log_model = LogisticRegression()
m = np.shape(X2)[0]

# 10-folds CV
y2_pred = cross_val_predict(log_model, X2, y2, cv=10)
print(metrics.accuracy_score(y2, y2_pred))

# LOOCV
# from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
accuracy = 0
for train, test in loo.split(X2):
    log_model.fit(X2[train], y2[train])  # fitting
    y2_p = log_model.predict(X2[test])
    if y2_p == y2[test]: accuracy += 1
print(accuracy / np.shape(X2)[0])

'''
针对经验风险最小化算法的过拟合的问题，给出交叉验证的方法，这个方法在做分类问题时很常用：
一：简单的交叉验证的步骤如下：
1、从全部的训练数据 S中随机选择 中随机选择 s的样例作为训练集train，剩余的作为测试集test。
2、通过对测试集训练 ，得到假设函数或者模型 。
3、在测试集对每一个样本根据假设函数或者模型，得到训练集的类标，求出分类正确率。
4，选择具有最大分类率的模型或者假设。
这种方法称为 hold -out cross validation 或者称为简单交叉验证。由于测试集和训练集是分开的，就避免了过拟合的现象

二：k折交叉验证 k-fold cross validation
1、将全部训练集 S分成 k个不相交的子集，假设 S中的训练样例个数为 m，那么每一个子 集有 m/k 个训练样例，，相应的子集称作 {s1,s2,…,sk}。
2、每次从分好的子集中里面，拿出一个作为测试集，其它k-1个作为训练集
3、根据训练训练出模型或者假设函数。
4、把这个模型放到测试集上，得到分类率。
5、计算k次求得的分类率的平均值，作为该模型或者假设函数的真实分类率。
这个方法充分利用了所有样本。但计算比较繁琐，需要训练k次，测试k次。

三：留一法  leave-one-out cross validation
留一法就是每次只留下一个样本做测试集，其它样本做训练集，如果有k个样本，则需要训练k次，测试k次。
留一法计算最繁琐，但样本利用率最高。适合于小样本的情况。

'''

'''
 两种交叉验证的结果相近，但是由于 Blood Transfusion Service Center Data Set的类分性不如iris明显，所得结果也要差一些。
 同时由程序运行可以看出，LOOCV的运行时间相对较长，这一点随着数据量的增大而愈发明显。
 所以，一般情况下选择K-折交叉验证即可满足精度要求，同时运算量相对小
'''
