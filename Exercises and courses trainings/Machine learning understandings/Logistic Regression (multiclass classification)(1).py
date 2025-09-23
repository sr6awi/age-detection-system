#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_digits
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
digits = load_digits()


# In[4]:


dir(digits)


# In[5]:


digits.data[0]


# In[16]:


plt.gray() 
for i in range(5):
    plt.matshow(digits.images[i]) 


# In[17]:


digits.target[0:5]


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.2)


# In[20]:


len(X_train)


# In[21]:


len(X_test)


# In[22]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[23]:


model.fit(X_train,y_train)


# In[24]:


model.score(X_test,y_test)


# In[25]:


plt.matshow(digits.images[56])


# In[26]:


digits.target[56]


# In[27]:


model.predict([digits.data[56]])


# In[29]:


model.predict(digits.data[0:5])


# In[31]:


y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_predicted)
cm


# In[32]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




