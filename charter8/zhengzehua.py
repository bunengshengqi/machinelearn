# 正则化 L1   L2
"""
首先看两个问题
1. L1、L2的提出是解决什么问题的？
L1和L2正则化的提出是为了解决线性回归中的过拟合问题。

过拟合指的是模型在训练集上表现很好，但在新的未见过的数据上表现较差的现象。当模型过于复杂或训练数据较少时，线性回归模型往往容易过拟合。过拟合的主要原因是模型过度拟合了训练数据中的噪声和异常值，导致在新的数据上的泛化能力较差。

L1和L2正则化通过在损失函数中引入模型参数的正则化项，限制了模型参数的大小，从而减少了过拟合的风险。

L1正则化（Lasso正则化）通过对模型参数的绝对值之和进行惩罚，倾向于将一些不重要的特征的权重变为零，实现特征选择和稀疏性。这样可以剔除对预测目标贡献较小的特征，降低模型的复杂性，提高模型的泛化能力。

L2正则化（岭回归）通过对模型参数的平方和进行惩罚，使得模型参数尽可能小，并对异常值具有一定的鲁棒性。这样可以减小模型参数的值，防止过拟合，提高模型的泛化能力。

总而言之，L1和L2正则化的提出旨在通过限制模型参数的大小，减少过拟合风险，并提高模型在未见过的数据上的表现。

2.线性回归算法中的正则，L1、L2.作用是什么？为什么要使用？使用的条件是啥？在sklearn中，如何使用？
在线性回归算法中，正则化是一种用于控制模型复杂度的技术。在正则化过程中，通过添加一个正则化项（也称为惩罚项）到损失函数中，来惩罚模型的复杂性。L1和L2正则化是两种常见的正则化方法。

L1正则化（Lasso正则化）通过在损失函数中添加模型参数的绝对值之和，即L1范数，来惩罚模型的复杂性。它的作用是使得模型参数中的一些特征的权重变为零，从而实现特征选择和稀疏性。L1正则化可以用于特征选择，因为它倾向于将不重要的特征的权重降低到零，从而减少了模型的复杂度。

L2正则化（岭回归）通过在损失函数中添加模型参数的平方和，即L2范数，来惩罚模型的复杂性。它的作用是使得模型参数尽可能小，并且对异常值具有一定的鲁棒性。L2正则化可以防止模型过拟合，并提高模型的泛化能力。

使用正则化的主要原因是为了解决过拟合问题。当训练数据集较小或特征维度较高时，模型容易过拟合，即在训练集上表现很好，但在测试集上表现较差。通过添加正则化项，可以限制模型的复杂性，减少过拟合的风险，并提高模型的泛化能力。

在scikit-learn（sklearn）库中，可以使用`linear_model`模块中的`Lasso`类来进行L1正则化，使用`Ridge`类来进行L2正则化。下面是使用示例：

```python
from sklearn.linear_model import Lasso, Ridge

# 使用L1正则化（Lasso）
lasso_model = Lasso(alpha=0.1)  # alpha是正则化强度，可根据需要调整
lasso_model.fit(X, y)  # X是特征矩阵，y是目标变量

# 使用L2正则化（Ridge）
ridge_model = Ridge(alpha=0.1)  # alpha是正则化强度，可根据需要调整
ridge_model.fit(X, y)  # X是特征矩阵，y是目标变量
```

在上述示例中，`alpha`参数控制正则化的强度，较大的`alpha`值会增加正则化的效果，较小的`alpha`值则减弱正则化的效果。你可以根据具体情况调整`alpha`的值来控制正则化的程度。

重点讲述l2就是岭回归的api
sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True,solver='auto',normalize=False)
alpha正则化
正则化力度越大，权重系数会越小
正则化力度越小，权重系数会越大

normalize：
默认对数据进行了标准化的处理
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, Ridge
from sklearn.metrics import mean_squared_error

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

if __name__ == '__main__':
    linear_model2()


"""
问题3：Ridge和RidgeCV的区别？
Ridge和RidgeCV都是用于进行岭回归（Ridge Regression）的方法，但它们之间有一些区别。

1. Ridge（岭回归）：
   - Ridge是用于进行岭回归的基本方法。
   - 它通过添加L2正则化（平方和惩罚项）来限制模型参数的大小。
   - 在使用Ridge进行建模时，需要手动选择正则化参数alpha的值。
   - Ridge的主要目标是减小模型的复杂性，防止过拟合。

2. RidgeCV（带交叉验证的岭回归）：
   - RidgeCV是Ridge的扩展版本，结合了交叉验证来选择最佳的正则化参数alpha的值。
   - 它通过在不同的alpha值上进行交叉验证，选择在验证集上表现最好的alpha值。
   - RidgeCV自动进行了交叉验证过程，不需要手动指定alpha的值。
   - RidgeCV的主要目标是在进行岭回归时，自动选择最佳的正则化参数alpha，以得到更好的模型性能。

总结：
- Ridge是基本的岭回归方法，需要手动选择正则化参数alpha的值。
- RidgeCV是带交叉验证的岭回归方法，自动选择最佳的正则化参数alpha的值，无需手动指定。

在实际应用中，如果你已经有一组合适的alpha值可以使用，或者对正则化参数的选择有特定要求，可以使用Ridge。而如果你希望通过交叉验证来自动选择最佳的alpha值，可以使用RidgeCV。

"""
# 以下展示了一个Ridgecv的例子，说明一下Ridgecv的使用方法和一些解释

# 下面是一个使用RidgeCV进行带交叉验证的岭回归的示例：

from sklearn.linear_model import RidgeCV
from sklearn.datasets import make_regression

# 创建一个模拟的回归数据集
X, y = make_regression(n_samples=100, n_features=10, random_state=42)

# 创建RidgeCV对象
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])

# 拟合数据
ridge_cv.fit(X, y)

# 打印最佳的alpha值
print("Best alpha:", ridge_cv.alpha_)

# 打印模型的系数
print("Coefficients:", ridge_cv.coef_)

"""
在上述示例中，首先使用`make_regression`函数创建了一个模拟的回归数据集。
然后，创建了一个`RidgeCV`对象并指定了一组候选的alpha值。接下来，调用`fit`方法拟合数据集，
RidgeCV会自动进行交叉验证来选择最佳的alpha值。最后，通过`alpha_`属性可以获取到最佳的alpha值，
通过`coef_`属性可以获取到模型的系数（权重）。

请注意，上述示例中的alpha值只是一个示例，你可以根据具体情况调整alpha值的范围和间隔。

"""




