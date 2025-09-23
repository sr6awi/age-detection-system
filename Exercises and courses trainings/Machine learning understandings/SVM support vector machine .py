#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris 
iris = load_iris()


# In[2]:


dir(iris)


# In[3]:


iris.feature_names


# In[5]:


df = pd.DataFrame(iris.data , columns = iris.feature_names)
df.head()


# In[6]:


df['target'] = iris.target
df.head()


# In[7]:


iris.target_names


# In[9]:


df[df.target==2].head()


# In[12]:


df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df.head()


# In[13]:


from matplotlib import pyplot as plt


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]


# In[18]:


df0.head()


# In[24]:


plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='blue',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='red',marker='.')


# In[25]:


plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='blue',marker='+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='red',marker='.')


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X = df.drop(['target','flower_name'],axis='columns')
X.head()


# In[28]:


y = df.target
y


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[30]:


len(X_train)


# In[31]:


len(X_test)


# In[52]:


from sklearn.svm import SVC
model  = SVC(gamma=100)


# In[53]:


model.fit(X_train,y_train)


# In[54]:


model.score(X_test,y_test)


# In[ ]:




