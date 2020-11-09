'''
FM(因子分解机)模型算法：稀疏数据下的特征二阶组合问题（个性化特征）
1、应用矩阵分解思想，引入隐向量构造FM模型方程
2、目标函数（损失函数复合FM模型方程）的最优问题：链式求偏导
3、SGD优化目标函数
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''二分类输出非线性映射'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
'''计算logit损失函数：L = log(1 + e**(y_hat * y))'''
def logit(y, y_hat):
    return np.log(1 + np.exp(-y * y_hat))
'''计算logit损失函数的外层偏导(不含y_hat的一阶偏导)'''
def df_logit(y, y_hat):
    return sigmoid(-y * y_hat) * (-y)
'''FM的模型方程：LR线性组合 + 交叉项组合 = 1阶特征组合 + 2阶特征组合'''
def FM(Xi, w0, W, V):
    # 样本Xi的特征分量xi和xj的2阶交叉项组合系数wij = xi和xj对应的隐向量Vi和Vj的内积
    # 向量形式：Wij:= <Vi, Vj> * Xi * Xj
    interaction = np.sum((Xi.dot(V)) ** 2 - (Xi ** 2).dot(V ** 2))  # 二值硬核匹配->向量软匹配
    y_hat = w0 + Xi.dot(W) + interaction / 2  # FM预测函数
    return y_hat[0]
'''SGD更新FM模型的参数列表：[w0, W, V]'''
def FM_SGD(X, y, k=2, alpha=0.01, iter=50):
    m, n = np.shape(X)
    w0, W = 0, np.zeros((n, 1))  # 初始化wo=R、W=(n, 1)
    V = np.random.normal(loc=0, scale=1, size=(n, k))  # 初始化隐向量矩阵V=(n, k)~N(0, 1)，其中Vj是第j维特征的隐向量
    all_FM_params = []  # FM模型的参数列表：[w0, W, V]
    for it in range(iter):
        total_loss = 0  # 当前迭代模型的损失值
        for i in range(m):  # 遍历训练集
            y_hat = FM(Xi=X[i], w0=w0, W=W, V=V)  # FM的模型方程
            total_loss += logit(y=y[i], y_hat=y_hat)  # 计算logit损失函数值
            dloss = df_logit(y=y[i], y_hat=y_hat)  # 计算logit损失函数的外层偏导
            dloss_w0 = dloss * 1  # l(y, y_hat)中y_hat展开w0，求关于w0的内层偏导
            w0 = w0 - alpha * dloss_w0  # 梯度下降更新w0
            for j in range(n):  # 遍历n维向量X[i]
                if X[i, j] != 0:
                    dloss_Wj = dloss * X[i, j]  # l(y, y_hat)中y_hat展开y_hat，求关于W[j]的内层偏导
                    W[j] = W[j] - alpha * dloss_Wj  # 梯度下降更新W[j]
                    for f in range(k):  # 遍历k维隐向量Vj
                        # l(y, y_hat)中y_hat展开V[j, f]，求关于V[j, f]的内层偏导
                        dloss_Vjf = dloss * (X[i, j] * (X[i].dot(V[:, f])) - V[j, f] * X[i, j] ** 2)
                        V[j, f] = V[j, f] - alpha * dloss_Vjf  # 梯度下降更新V[j, f]
        print('FM第{}次迭代，当前损失值为：{:.4f}'.format(it + 1, total_loss / m))
        all_FM_params.append([w0, W, V])  # 保存当前迭代下FM的参数列表:[w0, W, V]
    return all_FM_params
'''FM模型预测测试集分类结果'''
def FM_predict(X, w0, W, V):
    predicts, threshold = [], 0.5  # sigmoid阈值设置
    for i in range(X.shape[0]):  # 遍历测试集
        y_hat = FM(Xi=X[i], w0=w0, W=W, V=V)  # FM的模型方程
        predicts.append(-1 if sigmoid(y_hat) < threshold else 1)  # 分类结果非线性映射
    return np.array(predicts)
'''FM在不同迭代次数下的参数列表中，训练集的损失值和测试集的准确率变化'''
def draw_research(all_FM_params, X_train, y_train, X_test, y_test):
    all_total_loss, all_total_accuracy = [], []
    for w0, W, V in all_FM_params:
        total_loss = 0
        for i in range(X_train.shape[0]):
            total_loss += logit(y=y_train[i], y_hat=FM(Xi=X_train[i], w0=w0, W=W, V=V))
        all_total_loss.append(total_loss / X_train.shape[0])
        all_total_accuracy.append(accuracy_score(y_test, FM_predict(X=X_test, w0=w0, W=W, V=V)))
    plt.plot(np.arange(len(all_FM_params)), all_total_loss, color='#FF4040', label='训练集的损失值')
    plt.plot(np.arange(len(all_FM_params)), all_total_accuracy, color='#4876FF', label='测试集的准确率')
    plt.xlabel('SGD迭代次数')
    plt.title('FM模型:二阶互异特征组合')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    np.random.seed(123)
    df = pd.read_csv(r'D:\\FM-master\\data\\xg.csv')
    df['Class'] = df['Class'].map({0: -1, 1: 1})  # 标签列从[0, 1]离散到[-1, 1]
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1].values, df.iloc[:, -1].values, test_size=0.3, random_state=123)
    X_train = MinMaxScaler().fit_transform(X_train)  # 归一化训练集，返回[0, 1]区间
    X_test = MinMaxScaler().fit_transform(X_test)  # 归一化测试集，返回[0, 1]区间
    '''*****************FM预测模型*****************'''
    all_FM_params = FM_SGD(X=X_train, y=y_train, k=2, alpha=0.01, iter=45)  # SGD更新FM模型的参数列表：[w0, W, V]
    w0, W, V = all_FM_params[-1]  # FM模型的参数列表
    predicts = FM_predict(X=X_test, w0=w0, W=W, V=V)  # FM模型预测测试集分类结果 80.52%  80.09%
    print('FM在测试集的分类准确率为: {:.2%}'.format(accuracy_score(y_test, predicts)))
    # draw_research(all_FM_params=all_FM_params, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv('D:\\u.data', sep='\t', header=None, names=rnames, engine='python')
#构造2分类数据集

df['rating']=df['rating'].map(lambda x: -1 if x>=3 else 1) #1,2是label=1  3，4,5是label=0

#one-hot encoder
from sklearn.preprocessing import OneHotEncoder
columns=['user_id', 'movie_id']

for i in columns:
    get_dummy_feature=pd.get_dummies(df[i])
    df=pd.concat([df, get_dummy_feature],axis=1)
    df=df.drop(i, axis=1)

df=df.drop(['timestamp'], axis=1)
#这些特征可以进一步挖掘。这里都不要了，只保留one-hot特征

from sklearn.model_selection import train_test_split

X=df.drop('rating', axis=1)
Y=df['rating']

X_train,X_val,Y_train,Y_val=train_test_split(X, Y, test_size=0.3, random_state=123)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(y, y_hat): #对每一个样本计算损失
    if y_hat == 'nan':
        return 0
    else:
        return np.log(1 + np.exp(-y * y_hat))

def df_logit(y, y_hat):
    return sigmoid(-y * y_hat) * (-y)

from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter


class FactorizationMachine(BaseEstimator):
    def __init__(self, k=5, learning_rate=0.01, iternum=2):
        self.w0 = None
        self.W = None
        self.V = None
        self.k = k
        self.alpha = learning_rate
        self.iternum = iternum

    def _FM(self, Xi):
        interaction = np.sum((Xi.dot(self.V)) ** 2 - (Xi ** 2).dot(self.V ** 2))
        y_hat = self.w0 + Xi.dot(self.W) + interaction / 2
        return y_hat[0]

    def _FM_SGD(self, X, y):
        m, n = np.shape(X)
        # 初始化参数
        self.w0 = 0
        self.W = np.random.uniform(size=(n, 1))
        self.V = np.random.uniform(size=(n, self.k))  # Vj是第j个特征的隐向量  Vjf是第j个特征的隐向量表示中的第f维

        for it in range(self.iternum):
            total_loss = 0
            for i in range(m):  # 遍历训练集
                y_hat = self._FM(Xi=X[i])  # X[i]是第i个样本  X[i,j]是第i个样本的第j个特征

                total_loss += logit(y=y[i], y_hat=y_hat)  # 计算logit损失函数值
                dloss = df_logit(y=y[i], y_hat=y_hat)  # 计算logit损失函数的外层偏导

                dloss_w0 = dloss * 1  # 公式中的w0求导，计算复杂度O(1)
                self.w0 = self.w0 - self.alpha * dloss_w0

                for j in range(n):
                    if X[i, j] != 0:
                        dloss_Wj = dloss * X[i, j]  # 公式中的wi求导，计算复杂度O(n)
                        self.W[j] = self.W[j] - self.alpha * dloss_Wj
                        for f in range(self.k):  # 公式中的vif求导，计算复杂度O(kn)
                            dloss_Vjf = dloss * (X[i, j] * (X[i].dot(self.V[:, f])) - self.V[j, f] * X[i, j] ** 2)
                            self.V[j, f] = self.V[j, f] - self.alpha * dloss_Vjf

            print('iter={}, loss={:.4f}'.format(it+1, total_loss / m))

        return self

    def _FM_predict(self, X):
        predicts, threshold = [], 0.5  # sigmoid阈值设置
        for i in range(X.shape[0]):  # 遍历测试集
            y_hat = self._FM(Xi=X[i])  # FM的模型方程
            predicts.append(-1 if sigmoid(y_hat) < threshold else 1)
        return np.array(predicts)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = np.array(X)
            y = np.array(y)

        return self._FM_SGD(X, y)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = np.array(X)

        return self._FM_predict(X)

    def predict_proba(self, X):
        pass

from sklearn.metrics import roc_auc_score, confusion_matrix

model=FactorizationMachine(k=10, learning_rate=0.001, iternum=2)
model.fit(X_train, Y_train)

y_pred=model.predict(X_train)

print('训练集roc: {:.2%}'.format(roc_auc_score(Y_train.values, y_pred)))
print('混淆矩阵: \n',confusion_matrix(Y_train.values, y_pred))

y_true=model.predict(X_val)

print('验证集roc: {:.2%}'.format(roc_auc_score(Y_val.values, y_true)))
print('混淆矩阵: \n',confusion_matrix(Y_val.values, y_true))

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

X_val = MinMaxScaler().fit_transform(X_val)#归一化测试集，返回[0,1]区间

val_predicts = model._FM_predict(X_val)
print('FM测试集的分类准确率为: {:.2%}'.format(accuracy_score(Y_val,val_predicts)))
print("FM测试集均方误差mse：{:.2%}".format(mean_squared_error(Y_val,val_predicts)))
print("FM测试集召回率recall：{:.2%}".format(recall_score(Y_val,val_predicts)))
print("FM测试集的精度precision：{:.2%}".format(precision_score(Y_val,val_predicts)))
