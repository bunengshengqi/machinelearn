第九章的内容介绍
1.逻辑回归的介绍
2.逻辑回归的api介绍
3.案例讲解--肿瘤预测案例
4.分类评估方法介绍
5.ROC曲线的绘制过程
6.类别不平衡数据介绍（补充）
7.过采样和欠采样的介绍（补充）

学习目标：
逻辑回归的损失函数、优化方法；
逻辑回归的应用场景；
使用逻辑回归实现逻辑回归的预测
精确率、召回率等指标的区别
解决样本不均衡情况下的评估
绘制roc曲线图形

逻辑回归是分类模型，是解决二分类问题的利器，
逻辑回归的输入就是线性回归的输出，并且经过sigmoid函数，
输出[0,1]之间的概率值。默认0.5为阈值。



知识回顾：线性回归的损失函数和逻辑回归的损失函数
线性回归和逻辑回归都是常见的机器学习算法，它们使用不同的损失函数来进行模型训练。

1. 线性回归的损失函数（Mean Squared Error，均方误差）：
线性回归旨在拟合一个线性模型来预测实数值输出。它使用均方误差作为损失函数，表示预测值与真实值之间的平均差异。均方误差的数学形式如下：

```
MSE = (1/n) * Σ(y - ŷ)^2
```

其中，MSE是均方误差，n是样本数量，y是实际值，ŷ是线性回归模型的预测值。线性回归的目标是通过最小化均方误差来找到最佳拟合直线。

2. 逻辑回归的损失函数（Logistic Loss，对数损失或交叉熵）：
逻辑回归用于分类问题，它输出一个概率值来表示样本属于某个类别的概率。逻辑回归使用对数损失函数（也称为交叉熵）来度量模型预测概率与实际标签之间的差异。对数损失的数学形式如下：

```
LogLoss = -(1/n) * Σ[y * log(ŷ) + (1-y) * log(1-ŷ)]
```

其中，LogLoss是对数损失，n是样本数量，y是实际标签（0或1），ŷ是逻辑回归模型的预测概率。逻辑回归的目标是通过最小化对数损失来找到最佳的分类边界。

需要注意的是，以上是线性回归和逻辑回归的常见损失函数形式，具体实现中可能会有一些变体或调整，但基本思想和数学原理是相似的。


准确率、精确率、召回率、F1-score、ROC曲线、AUC指标
准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1-score、ROC曲线和AUC指标都是用于评估分类模型性能的指标，每个指标关注不同的方面。下面我将对它们进行逐个解释：

1. 准确率（Accuracy）：
准确率是分类模型预测正确的样本数量与总样本数量之比。它衡量了模型的整体预测准确性，计算公式为：

```
Accuracy = (预测正确的样本数) / (总样本数)
```

准确率越高，模型的整体预测准确性越好。

2. 精确率（Precision）：
精确率是分类模型预测为正类的样本中真正为正类的比例。它衡量了模型在预测为正类的样本中的准确性，计算公式为：

```
Precision = (真正为正类的样本数) / (预测为正类的样本数)
```

精确率较高表示模型在预测为正类的样本中具有较高的准确性。

3. 召回率（Recall）：
召回率是分类模型预测为正类的样本中真正为正类的比例。它衡量了模型对真正为正类的样本的预测能力，计算公式为：

```
Recall = (真正为正类的样本数) / (真实为正类的样本数)
```

召回率较高表示模型能够较好地捕捉到真正为正类的样本。

4. F1-score：
F1-score是精确率和召回率的调和平均值，综合了模型的准确性和预测能力。它计算公式为：

```
F1-score = 2 * (Precision * Recall) / (Precision + Recall)
```

F1-score综合了精确率和召回率的性能，适用于评估模型的综合性能。

5. ROC曲线（Receiver Operating Characteristic curve）：
ROC曲线是一种用于可视化二分类模型性能的图形。它以不同的分类阈值为基准，绘制了模型的真正例率（True Positive Rate，召回率）与假正例率（False Positive Rate）之间的关系曲线。ROC曲线越靠近左上角，表示模型性能越好。

6. AUC指标（Area Under the ROC Curve）：
AUC指标是ROC曲线下的面积，用于衡量模型在不同分类阈值下的整体性能。AUC的取值范围在0到1之间，越接近1表示模型的性能越好。AUC是评估模型性能的重要指标之一

（***）样本不均衡的问题：若两个类别，一个占4份，一个占1份，或者超过这个，一个5份，一个1份，就认为样本不均衡，导致所有的指标失效，需要引入ROC曲线,AUC指标

总结：
1 混淆矩阵
真正例（TP）
伪反例（FN）
伪正例（FP）
真反例（TN）
2.精确率与召回率
准确率：（对不对）（TP+TN）/(TP+TN+FN+FP)
精确率：（查的准不准）TP/(TP+FP)
召回率：（查的全不全）TP/(TP+FN)
F1-score:反应模型的稳健性
3.ROC曲线和AUC指标
ROC曲线：通过tpr和fpr，来进行图形绘制，绘制以后形成一个指标AUC
AUC指标：
越接近1，效果越好，越接近0，效果越差，越接近0.5，效果就是胡扯
注意：
这个指标主要用于评价不平衡的二分类问题。


3.5 ROC曲线绘制
小结
1.构建模型，把模型的概率值从大到小进行排序
2.从刚概率最大的点开始取值，一直进行tpr和fpr的计算，然后构建着整体模型，得到结果
3.其实就是在求解积分

ROC曲线（Receiver Operating Characteristic curve）是用于评估分类模型性能的一种常见方法，特别适用于二分类问题。ROC曲线显示了模型在不同阈值下的真正例率（True Positive Rate，TPR）和假正例率（False Positive Rate，FPR）之间的关系。

下面是绘制ROC曲线的一般步骤：

1. 收集模型预测结果和真实标签：首先，需要获取模型对一组样本的预测结果以及这些样本的真实标签（真实值）。通常，模型的预测结果可以是概率值或者分类标签。

2. 计算TPR和FPR：使用不同的阈值，根据模型的预测结果将样本分类为正例或负例。然后，计算出每个阈值下的TPR和FPR。TPR表示正例中被正确分类的比例，计算公式为TPR = TP / (TP + FN)，其中TP表示真正例数，FN表示假反例数。FPR表示负例中被错误分类的比例，计算公式为FPR = FP / (FP + TN)，其中FP表示假正例数，TN表示真反例数。

3. 绘制ROC曲线：将计算得到的不同阈值下的TPR和FPR值绘制成坐标系上的点，连接这些点即可得到ROC曲线。通常，横轴表示FPR，纵轴表示TPR。

4. 计算AUC值：计算ROC曲线下的面积（Area Under the Curve，AUC），AUC可以用来量化模型的性能。AUC的取值范围在0到1之间，AUC值越接近1，表示模型性能越好。

以下是一个简单的例子，说明如何绘制ROC曲线：

假设有一个二分类模型，对一组样本进行分类，并给出了如下的预测结果和真实标签：

| 预测结果 | 真实标签 |
| -------- | -------- |
| 0.9      | 正例     |
| 0.7      | 负例     |
| 0.6      | 正例     |
| 0.4      | 负例     |
| 0.3      | 正例     |
| 0.1      | 正例     |

根据以上预测结果和真实标签，可以计算出不同阈值下的TPR和FPR，例如：

- 当阈值为0.5时，有2个真正例（TP）、1个假正例（FP），所以TPR为2/3，FPR为1/2。
- 当阈值为0.3时，有3个真正例（TP）、2个假正例（FP），所以TPR为3/3，FPR为2/2。
- 当阈值为0.1时，有4个真正例

（TP）、3个假正例（FP），所以TPR为4/3，FPR为3/2。

根据以上计算结果，可以绘制出ROC曲线，连接不同阈值下的TPR和FPR点，最终得到模型的ROC曲线。根据ROC曲线的形状以及计算AUC值，可以评估模型的分类性能。

转载链接，如有侵权，请告知！删除
https://blog.csdn.net/weixin_42462804/article/details/100015334

