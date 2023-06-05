"""
---流程分析（机器学习的流程）
# 1.获取数据
# 2.数据基本处理
# 2.1 数据分割
# 3.特征工程--标准化
# 4.机器学习--线性回归
# 5.模型评估
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, Ridge
from sklearn.metrics import mean_squared_error


def linear_model():
    """
    线性回归：基于正规方程
    :return:
    """
    # 1.获取数据
    boston = load_boston()
    # 2.数据基本处理
    # 2.1 数据分割
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
    # 3.特征工程--标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4.机器学习--线性回归
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    print("这个模型的偏置是：\n", estimator.intercept_)
    print("这个模型的系数是：\n", estimator.coef_)
    # 5.模型评估
    # 5.1 预测值
    y_pre = estimator.predict(x_test)
    print("这个模型的预测值是：\n", y_pre)
    # 5.2 均方误差
    ret = mean_squared_error(y_test, y_pre)
    print("这个模型的均方误差是：\n", ret)

def linear_model1():
    """
    线性回归：梯度下降法
    :return:
    """
    # 1.获取数据
    boston = load_boston()
    # 2.数据基本处理
    # 2.1 数据分割
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
    # 3.特征工程--标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4.机器学习--线性回归
    # estimator = SGDRegressor(max_iter=1000, learning_rate="constant", eta0=1)
    estimator = SGDRegressor(max_iter=1000)
    estimator.fit(x_train, y_train)

    print("这个模型的偏置是：\n", estimator.intercept_)
    print("这个模型的系数是：\n", estimator.coef_)
    # 5.模型评估
    # 5.1 预测值
    y_pre = estimator.predict(x_test)
    print("这个模型的预测值是：\n", y_pre)
    # 5.2 均方误差
    ret = mean_squared_error(y_test, y_pre)
    print("这个模型的均方误差是：\n", ret)


def linear_model2():
    """
    线性回归：岭回归
    :return:NONE
    """
    # 1.获取数据
    boston = load_boston()
    # 2.数据基本处理
    # 2.1 数据分割
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
    # 3.特征工程--标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4.机器学习--线性回归
    # estimator = Ridge(alpha=1.0)
    estimator = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0, 10, 100)) # 交叉验证
    estimator.fit(x_train, y_train)

    print("这个模型的偏置是：\n", estimator.intercept_)
    print("这个模型的系数是：\n", estimator.coef_)
    # 5.模型评估
    # 5.1 预测值
    y_pre = estimator.predict(x_test)
    print("这个模型的预测值是：\n", y_pre)
    # 5.2 均方误差
    ret = mean_squared_error(y_test, y_pre)
    print("这个模型的均方误差是：\n", ret)




# linear_model()
# linear_model1()
linear_model2()