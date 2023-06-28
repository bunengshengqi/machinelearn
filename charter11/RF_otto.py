#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# 数据处理


# In[5]:


data = pd.read_csv('./ottotrain.csv')


# In[6]:


data.head()


# In[7]:


data.shape


# In[8]:


data.describe()


# In[9]:


# 通过图形可视化，查看数据分布的情况
import seaborn as sns
sns.countplot(data.target)
plt.show()


# # 由上图可知，该数据类别不均衡，所以需要后期处理

# In[11]:


data.target


# #  2 数据基本处理
# 数据已经做过脱敏，不在需要特殊处理

# ##   2.1由于数据较大，这里采用截取部分数据的方式

# In[52]:


new1_data = data[:1000]


# In[53]:


new1_data


# In[54]:


new1_data.shape


# In[55]:


# 图形可视化，查看数据分布
import seaborn as sns
sns.countplot(new1_data.target)
plt.show()


# 使用上述方式获取不可行，然后使用随机欠采样获取相应的数据

# In[ ]:


# 随机欠采样获取数据,在欠采样之前，首先确定特征值/标签值。所以第一步是
# 第一步：首先需要确定特征值/标签值
y = data['target']
x = data.drop(['id','target'],axis=1)  #按照列删除  axis=1


# In[56]:


x.head()


# In[57]:


y.head()


# In[59]:


# 欠采样获取数据
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled,Y_resampled = rus.fit_resample(x,y)


# In[60]:


x.shape,y.shape


# In[61]:


X_resampled.shape,Y_resampled.shape


# In[62]:


# 图形可视化，查看数据分布情况
import seaborn as sns
sns.countplot(y_resampled)
plt.show()


# ## 2.2把标签值转换为数字

# In[67]:


Y_resampled.head()


# In[68]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_resampled = le.fit_transform(Y_resampled)


# In[69]:


Y_resampled


# ## 2.3 分割数据

# In[75]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(X_resampled, Y_resampled,test_size=0.2)


# In[76]:


x_test.shape,y_test.shape


# # 模型训练
# ## 基本模型训练

# In[77]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(oob_score=True) # 带外估计
rf.fit(x_train, y_train)


# In[79]:


y_pre = rf.predict(x_test)
y_pre


# In[80]:


rf.oob_score_


# In[82]:


# 图形可视化，查看数据分布
import seaborn as sns
sns.countplot(y_pre)
plt.show()


# In[83]:


# logloss 模型评估
from sklearn.metrics import log_loss
log_loss(y_test,y_pre,eps=1e-15,normalize=True)


# In[84]:


y_test,y_pre


# 上述报错原因：logloss使用规程中必须要求将输出使用onehot输出，
# 需要将这个多类别问题的输出结果需要通过oneHoeEncoder修改为如下：

# In[85]:


from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(sparse=False)

y_test1 = one_hot.fit_transform(y_test.reshape(-1,1))
y_pre1 = one_hot.fit_transform(y_pre.reshape(-1, 1))


# In[86]:


y_test1


# In[87]:


y_pre1


# In[89]:


# logloss 模型评估
log_loss(y_test1, y_pre1, eps=1e-15, normalize=True)


# In[90]:


# 改变预测值的输出，让输出为百分占比，降低logloss的值
y_pre_proba = rf.predict_proba(x_test)


# In[91]:


y_pre_proba


# In[92]:


rf.oob_score_


# In[93]:


# logloss 模型评估
log_loss(y_test1, y_pre_proba, eps=1e-15, normalize=True)


# ## 3.2 模型调优

# n_estimators, max_feature, max_depth, min_samples_leaf

# ### 3.2.1 确定最优的n_estimators

# In[95]:


# 确定n_esimators的取值范围
tuned_parameters = range(10, 200, 10)

# 创建添加accuracy的一个numpy
accuracy_t = np.zeros(len(tuned_parameters))

# 创建添加error的一个numpy
error_t = np.zeros(len(tuned_parameters))

# 调优过程实现
for j, one_parameter in enumerate(tuned_parameters):
    rf2 = RandomForestClassifier(n_estimators=one_parameter, 
                                 max_depth=10, 
                                 max_features=10, 
                                 min_samples_leaf=10, 
                                 oob_score=True, 
                                 random_state=0, 
                                 n_jobs=-1)
    rf2.fit(x_train, y_train)
    # 输出accuracy
    accuracy_t[j] = rf2.oob_score_
    
    # 输出log_loss
    y_pre = rf2.predict_proba(x_test)
    error_t[j] = log_loss(y_test, y_pre, eps=1e-15, normalize=True)
    
    print(error_t)


# In[96]:


# 优化结果可视化

fig, axes = plt.subplots(nrows=1, ncols=2, 
                         figsize=(20,4), dpi=100)
axes[0].plot(tuned_parameters, error_t)
axes[1].plot(tuned_parameters, accuracy_t)

axes[0].set_xlabel('n_estimators')
axes[0].set_ylabel('error_t')
axes[1].set_xlabel('n_estimators')
axes[1].set_ylabel('accuracy_t')

axes[0].grid(True)
axes[1].grid(True)

plt.show()


# 经过图像展示，最后确定n_estimators=175的时候，表现效果不错

# ### 3.2.2确定最优的 max_features

# In[97]:


# 确定max_features的取值范围
tuned_parameters = range(5, 40, 5)

# 创建添加accuracy的一个numpy
accuracy_t = np.zeros(len(tuned_parameters))

# 创建添加error的一个numpy
error_t = np.zeros(len(tuned_parameters))

# 调优过程实现
for j, one_parameter in enumerate(tuned_parameters):
    rf2 = RandomForestClassifier(n_estimators=175, 
                                 max_depth=10, 
                                 max_features=one_parameter, 
                                 min_samples_leaf=10, 
                                 oob_score=True, 
                                 random_state=0, 
                                 n_jobs=-1)
    rf2.fit(x_train, y_train)
    # 输出accuracy
    accuracy_t[j] = rf2.oob_score_
    
    # 输出log_loss
    y_pre = rf2.predict_proba(x_test)
    error_t[j] = log_loss(y_test, y_pre, eps=1e-15, normalize=True)
    
    print(error_t)


# In[98]:


# 优化结果可视化

fig, axes = plt.subplots(nrows=1, ncols=2, 
                         figsize=(20,4), dpi=100)
axes[0].plot(tuned_parameters, error_t)
axes[1].plot(tuned_parameters, accuracy_t)

axes[0].set_xlabel('max_features')
axes[0].set_ylabel('error_t')
axes[1].set_xlabel('max_features')
axes[1].set_ylabel('accuracy_t')

axes[0].grid(True)
axes[1].grid(True)

plt.show()


# 经过图像展示，最后确定max_features=15的时候，表现效果不错

# ### 3.2.3 确定最优的max_depth

# In[99]:


# 确定max_depth的取值范围
tuned_parameters = range(10, 100, 10)

# 创建添加accuracy的一个numpy
accuracy_t = np.zeros(len(tuned_parameters))

# 创建添加error的一个numpy
error_t = np.zeros(len(tuned_parameters))

# 调优过程实现
for j, one_parameter in enumerate(tuned_parameters):
    rf2 = RandomForestClassifier(n_estimators=175, 
                                 max_depth=one_parameter, 
                                 max_features=15, 
                                 min_samples_leaf=10, 
                                 oob_score=True, 
                                 random_state=0, 
                                 n_jobs=-1)
    rf2.fit(x_train, y_train)
    # 输出accuracy
    accuracy_t[j] = rf2.oob_score_
    
    # 输出log_loss
    y_pre = rf2.predict_proba(x_test)
    error_t[j] = log_loss(y_test, y_pre, eps=1e-15, normalize=True)
    
    print(error_t)


# In[100]:


# 优化结果可视化

fig, axes = plt.subplots(nrows=1, ncols=2, 
                         figsize=(20,4), dpi=100)
axes[0].plot(tuned_parameters, error_t)
axes[1].plot(tuned_parameters, accuracy_t)

axes[0].set_xlabel('max_depth')
axes[0].set_ylabel('error_t')
axes[1].set_xlabel('max_depth')
axes[1].set_ylabel('accuracy_t')

axes[0].grid(True)
axes[1].grid(True)

plt.show()


# 经过图像展示，最后确定max_depth=30的时候，表现效果不错

# ### 3.2.4 确定最优的min_samples_leaf

# In[101]:


# 确定min_samples_leaf的取值范围
tuned_parameters = range(1, 10, 2)

# 创建添加accuracy的一个numpy
accuracy_t = np.zeros(len(tuned_parameters))

# 创建添加error的一个numpy
error_t = np.zeros(len(tuned_parameters))

# 调优过程实现
for j, one_parameter in enumerate(tuned_parameters):
    rf2 = RandomForestClassifier(n_estimators=175, 
                                 max_depth=30, 
                                 max_features=15, 
                                 min_samples_leaf=one_parameter, 
                                 oob_score=True, 
                                 random_state=0, 
                                 n_jobs=-1)
    rf2.fit(x_train, y_train)
    # 输出accuracy
    accuracy_t[j] = rf2.oob_score_
    
    # 输出log_loss
    y_pre = rf2.predict_proba(x_test)
    error_t[j] = log_loss(y_test, y_pre, eps=1e-15, normalize=True)
    
    print(error_t)


# In[102]:


# 优化结果可视化

fig, axes = plt.subplots(nrows=1, ncols=2, 
                         figsize=(20,4), dpi=100)
axes[0].plot(tuned_parameters, error_t)
axes[1].plot(tuned_parameters, accuracy_t)

axes[0].set_xlabel('min_samples_leaf')
axes[0].set_ylabel('error_t')
axes[1].set_xlabel('min_samples_leaf')
axes[1].set_ylabel('accuracy_t')

axes[0].grid(True)
axes[1].grid(True)

plt.show()


# 经过图像展示，最后确定min_samples_leaf=1的时候，表现效果不错

# ## 3.3 确定最优模型
# n_estimators=175, 
# max_depth=30, 
# max_features=15, 
# min_samples_leaf=1

# In[103]:


rf3 = RandomForestClassifier(n_estimators=175,
                            max_depth=30,
                            max_features=15,
                            min_samples_leaf=1,
                            oob_score=True,
                            random_state=40,
                            n_jobs=-1)
rf3.fit(x_train, y_train)


# In[104]:


rf3.score(x_test, y_test)


# In[105]:


rf3.oob_score_


# In[106]:


y_pre_proba1 = rf3.predict_proba(x_test)
log_loss(y_test, y_pre_proba1)


# In[107]:


###  调优  网格化搜索调优，也可以像上述那样调优（多说一句）


# # 4 生成移交数据

# In[108]:


test_data = pd.read_csv('./ottotest.csv')


# In[109]:


test_data.head()


# In[110]:


test_data_drop_id = test_data.drop(["id"], axis=1)
test_data_drop_id.head()


# In[111]:


y_pre_test = rf3.predict_proba(test_data_drop_id)
y_pre_test


# In[112]:


resule_data = pd.DataFrame(y_pre_test, columns=["Class_" + str(i) for i in range(1, 10)])
resule_data.head()


# In[114]:


resule_data.insert(loc=0, column="id", value=test_data.id)
resule_data.head()


# In[115]:


resule_data.to_csv('./submission.csv')


# In[ ]:


"""

参考链接：https://zsyll.blog.csdn.net/article/details/119006636?ydreferer=aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2MDkyMDYxL2NhdGVnb3J5XzEwOTkzNTI1Lmh0bWw%3D

"""

