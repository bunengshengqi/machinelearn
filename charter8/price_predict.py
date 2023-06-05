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


"""
当编写机器学习模型的代码时，损失函数和模型评估是两个重要的概念。它们在训练和评估模型的过程中发挥不同的作用。

1. 损失函数：
   - 定义：损失函数是用来度量模型预测结果与真实标签之间的差异的函数。它衡量了模型在训练数据上的性能，并提供了一个可优化的目标函数。
   - 作用：在训练过程中，损失函数用于指导优化算法调整模型的参数，以使模型的预测结果尽可能接近真实标签。通过最小化损失函数，模型能够逐步改善在训练数据上的性能。

2. 模型评估：
   - 定义：模型评估是在训练完成后对模型性能进行评估的过程。它通过使用一些指标来衡量模型在未见过的数据上的表现。
   - 作用：模型评估的目的是了解模型的泛化能力，即模型在实际应用中的预测能力。通过评估指标，可以评估模型的准确率、精确率、召回率、F1值等，并判断模型是否达到了预期的性能要求。

在训练过程中，通常会通过以下步骤进行：
1. 定义模型的结构和参数。
2. 定义损失函数，根据任务类型选择适当的损失函数，如平方损失函数（用于回归问题）或交叉熵损失函数（用于分类问题）。
3. 使用训练数据输入模型，计算预测结果，并根据损失函数计算预测结果与真实标签之间的差异。
4. 根据损失函数的值，使用优化算法（如梯度下降）调整模型的参数，以最小化损失函数。
5. 重复以上步骤，直到达到一定的停止条件（如达到最大迭代次数或损失函数收敛）。

在模型训练完成后，可以进行模型评估的步骤：
1. 准备未见过的测试数据集。
2. 使用训练好的模型对测试数据进行预测。
3. 使用评估指标（如准确率、精确率、召回率等）来评估模型在测试数据上的性能。
4. 根据评估结果判断模型的泛化能力和是否满足预期要求。

总之，损失函数和模型评估在机器学习模型的训练和评估过程中起着不同的作用。
"""