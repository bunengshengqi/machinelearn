#!/usr/bin/env python
# coding: utf-8

# In[2]:


#1.获取数据
#2.基本数据处理
#2.1 缺失值处理
#2.2 确定特征值、目标值
#2.3 分割数据
#3.特征工程（标准化）
#4.机器学习（逻辑回归）
#5.模型评估


# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 准确率，精确率，召回率
from sklearn.metrics import classification_report, roc_auc_score

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# In[14]:


#names = ['样本编号','团块厚度','细胞大小的均匀性','细胞形状的均匀性','边缘附着力','单层上皮细胞大小','裸核','乏味染色质','正常核仁','线粒体','类别']

names = ['Sample ID','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli',
        'Mitoses','Class']

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=names)


# In[15]:


data.head()


# In[16]:


#2.基本数据处理
#2.1 缺失值处理
data = data.replace(to_replace="?", value=np.nan)
data = data.dropna()


# In[17]:


#2.2 确定特征值、目标值
x = data.iloc[:, 1:-1]
x.head()


# In[18]:


y = data['Class']
y.head()


# In[19]:


#2.3 分割数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=22)


# In[20]:


#3.特征工程（标准化）
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)


# In[21]:


#4.机器学习（逻辑回归）
estimator = LogisticRegression()
estimator.fit(x_train,y_train)


# In[22]:


#5.模型评估
# 5.1准确率
ret = estimator.score(x_test,y_test)
print("准群率为：\n",ret)
# 5.2预测值
y_pre = estimator.predict(x_test)
print("预测值：\n",y_pre)


# In[23]:


# 上述的例子中，准确率并不是最关心的！！！！最关心的是，在关注的样本中，
# 患病的人有没有被全部预测出来。

# 经验分享：
# 数据中如果存在缺失值，一定要进行处理
# 准确率并不是衡量分类正确的唯一标准


# In[24]:


# 5.3 精确率、召回率评价指标
ret = classification_report(y_test, y_pre, labels=(2,4),target_names=["良性","恶性"])
print(ret)


# In[25]:


# 5.4 AUC指标计算
y_test = np.where(y_test>3,1, 0)
roc_auc_score(y_test, y_pre)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




