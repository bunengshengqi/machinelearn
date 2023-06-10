#!/usr/bin/env python
# coding: utf-8

# In[14]:


from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

from collections import Counter


# In[15]:


x,y = make_classification(n_samples=5000,n_features=2,
                         n_informative=2,n_redundant=0,
                         n_repeated=0,n_classes=3,
                         n_clusters_per_class=1,
                         weights=[0.01,0.05,0.94],random_state=0)


# In[16]:


x


# In[17]:


x.shape


# In[18]:


y,y.shape


# In[19]:


Counter(y)


# In[20]:


#数据可视化
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()


# In[21]:


# 以上是准备类别不平衡数据，解决这个的两种方案：
# 1 过采样方法   减少数量较多那一类样本的数量，使得正负样本比例均衡
# 2 欠采样方法   增加数量较少那一类样本的数量，使得正负样本比例均衡


# In[24]:


# 2 过采样
# 2.1 随机过采样


# In[25]:


get_ipython().system('pip install imbalanced-learn')


# In[26]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x_resampled, y_resampled = ros.fit_resample(x,y)


# In[27]:


Counter(y_resampled)


# In[28]:


# 数据可视化
plt.scatter(x_resampled[:,0], x_resampled[:,1],c=y_resampled)
plt.show()


# In[ ]:


#  随机过采样的缺点。
# SMOTE过采样


# In[29]:


from imblearn.over_sampling import SMOTE
x_resampled,y_resampled = SMOTE().fit_resample(x,y)


# In[30]:


Counter(y_resampled)


# In[31]:


# 数据可视化
plt.scatter(x_resampled[:,0], x_resampled[:,1],c=y_resampled)
plt.show()


# In[ ]:


# 3 欠采样方法


# In[32]:


from imblearn.under_sampling import RandomUnderSampler
rus = RandomOverSampler(random_state=0)
x_resampled,y_resampled = rus.fit_resample(x, y)


# In[33]:


Counter(y_resampled)


# In[34]:


# 数据可视化
plt.scatter(x_resampled[:,0], x_resampled[:,1],c=y_resampled)
plt.show()


# In[ ]:





# In[ ]:




