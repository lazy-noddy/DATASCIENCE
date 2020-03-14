#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# # read dataset

# In[2]:


train=pd.read_csv('titanic_train.csv')


# In[3]:


train


# # Find out Missing Data

# In[4]:


train.isnull().sum()


# # Find out how many people survived / not survived

# In[5]:


train['Survived'].value_counts()


# In[6]:


train['Survived'].value_counts().plot.bar()


# # heatmap

# In[7]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# # remove column

# In[8]:


train.drop('Cabin',axis=1,inplace=True)


# In[9]:


train


# In[10]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# # Insert values to NAN in age column

# In[ ]:





# In[11]:


train['Age'].fillna(0)


# In[12]:


train.plot.bar(x='Pclass', y='Age') #use of plot.bar is not suitable for this data so lets try countplot from seaborn


# In[13]:


sns.countplot(x='Pclass',hue='Age',data=train) #use of countplot is not properly able to visualse so lets try boxplot


# In[14]:


sns.boxplot(x='Pclass',y='Age',data=train)


# In[15]:


train.describe() #get stats for whole table


# In[16]:


train['Age'].describe() # get stats for only one column


# In[17]:


def inp_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age    


# # to apply function use APPLY()

# In[18]:


train['Age']=train[['Age','Pclass']].apply(inp_age,axis=1)


# In[19]:


train['Age'].isnull().sum()


# In[20]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# # hence data is cleaned

# # find out how many male and female survived / not survived

# In[21]:


fm=train.groupby('Sex')


# In[22]:


fm['Survived'].value_counts()


# In[23]:


fm['Survived'].value_counts().plot.bar()


# # find out which class people survived / not survived

# In[24]:


cl=train.groupby('Pclass')


# In[25]:


cl['Survived'].value_counts()


# In[26]:


cl['Survived'].value_counts().plot.bar()


# # find avg age of people present in titanic

# In[27]:


train['Age'].mean()


# In[28]:


sns.distplot(train['Age'].dropna(),color='darkred',bins=40)


# In[29]:


train['Age'].hist(bins=30,color='darkred',alpha=0.6)


# # aplha use to set transparency

# # find out how many people having siblings  & spous

# In[30]:


train['SibSp'].value_counts()


# In[31]:


train['SibSp'].value_counts().plot.bar()


# # find the avg fare of tickets

# In[32]:


train['Fare'].mean()


# In[33]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[ ]:





# In[34]:


dummy=pd.get_dummies(train['Embarked'])
dummy.head()


# In[35]:


con=pd.concat([train,dummy],axis=1)
con.head()


# In[36]:


con.drop(columns=['Embarked'])
con.head()


# In[37]:


male_female=pd.get_dummies(con['Sex'])


# In[38]:


fm=pd.concat([con,male_female],axis=1).head()
main=fm.drop(columns=['Sex'])


# In[39]:


main


# In[40]:


#grouped=train.groupby('Pclass')


# In[41]:


#for cl,det in grouped:
    #mean=train[train['Pclass']==cl]['Age'].mean()
    #train.loc[grouped.get_group(cl).index,'Age']=mean


# In[42]:


#sns.heatmap(train.isnull(),cbar=False)


# In[43]:


train['Age']


# In[ ]:




