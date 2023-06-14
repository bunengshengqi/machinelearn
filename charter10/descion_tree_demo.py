#!/usr/bin/env python
# coding: utf-8

# In[81]:


# 1.获取数据
# 2.数据基本处理
# 2.1确定特征值、目标值
# 2.2缺失值处理
# 2.3数据集划分
# 3.特征工程（字典特征抽取）
# 4.机器学习（决策树算法）
# 5.模型评估


# In[82]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier,export_graphviz


# In[83]:


# 1.获取数据
titan = pd.read_csv('titanicttrain.csv')


# In[84]:


titan


# In[85]:


titan.describe()


# In[86]:


# 2.数据基本处理
# 2.1确定特征值、目标值
x = titan[["Pclass","Age","Sex"]]
y = titan['Survived']


# In[87]:


x.head()


# In[88]:


y.head()


# In[89]:


# 2.2缺失值处理
x["Age"].fillna(value=titan["Age"].mean(),inplace=True)


# In[90]:


x.head()


# In[91]:


# 2.3数据集划分
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=22,test_size=0.2)


# In[92]:


# 3.特征工程（字典特征抽取）
x_train.head()


# In[93]:


x_train = x_train.to_dict(orient="records")
x_test = x_test.to_dict(orient="records")


# In[94]:


x_train


# In[95]:


transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)


# In[96]:


x_test


# In[97]:


# 4.机器学习（决策树算法）


# In[98]:


estimator = DecisionTreeClassifier(max_depth=5) # 这里的max_depth可以根据数据集的大小指定
estimator.fit(x_train,y_train)


# In[99]:


# 5.模型评估
y_pre = estimator.predict(x_test)


# In[100]:


y_pre


# In[103]:


ret = estimator.score(x_test,y_test) #'Sex': 'female'
print(ret)


# In[105]:


export_graphviz(estimator,out_file="tree.dot",feature_names=transfer.get_feature_names_out())


# In[106]:


transfer.get_feature_names_out()


# In[107]:


export_graphviz(estimator,out_file="tree.dot",feature_names=['Age','Pclass','Sex=female', 'Sex=male'])


# In[ ]:


# https://blog.csdn.net/qq_46092061/article/details/118824940   参考链接，如侵权，联系我，秒删


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




