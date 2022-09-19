#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv('C:/Users/Aakash/Desktop/Iris.csv')


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.drop('Id', inplace=True, axis=1)


# In[9]:


df


# In[10]:


sns.pairplot(df,hue='Species')


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X = df.drop('Species',axis=1)


# In[13]:


X


# In[14]:


y = df['Species']


# In[15]:


y


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[18]:


dtree = DecisionTreeClassifier()


# In[19]:


dtree.fit(X_train,y_train)


# In[20]:


prediction = dtree.predict(X_test)


# In[21]:


from sklearn.metrics import classification_report, confusion_matrix


# In[22]:


print(confusion_matrix(y_test,prediction))
print('\n')
print(classification_report(y_test,prediction))


# In[23]:


print(confusion_matrix(y_test,prediction))


# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


rfc = RandomForestClassifier(n_estimators=200)


# In[26]:


rfc.fit(X_train,y_train)


# In[27]:


rfc_pred = rfc.predict(X_test)


# In[28]:


print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))


# In[29]:


print(confusion_matrix(y_test,rfc_pred))


# In[30]:


df['Species'].value_counts()


# In[31]:


df


# In[32]:


# Import necessary libraries for graph viz
from six import StringIO 
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

features = list(df.columns[:4])


# In[33]:


features


# In[34]:


dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




