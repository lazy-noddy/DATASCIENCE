#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


# In[4]:


data= load_breast_cancer()


# In[5]:


data


# In[13]:


breast_cancer=pd.DataFrame(data.data,columns=data.feature_names)
breast_cancer.head()


# In[18]:


breast_cancer['target'] = pd.Series(data.target)
breast_cancer.head()


# In[19]:


breast_cancer.tail()


# In[20]:


breast_cancer.info()


# In[22]:


x = breast_cancer.iloc[:,:-1].values
x


# In[24]:


y=breast_cancer.iloc[:,-1].values
y


# In[26]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[27]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[28]:


from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)


# In[29]:


nbpred = classifier.predict(x_test)
nbpred


# In[30]:


print(x_test[:10])
print('-'*15)
print(nbpred[:10])


# In[31]:


print(nbpred[:20])
print(y_test[:20])


# In[32]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,nbpred)
cm


# In[33]:


from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_test,nbpred))


# # naive bayes=91%

# # DecisionTree

# In[36]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state = 0) 
classifier.fit(x_train, y_train)


# In[37]:


dtpred = classifier.predict(x_test)
dtpred
print('-'*30)
print(x_test[:10])
print('-'*30)
print(dtpred[:10])
print('-'*30)
print(dtpred[:20])
print(y_test[:20])


# In[38]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,dtpred)
cm


# In[39]:


from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_test,dtpred))


# # Decision tree 95%

# # KNN

# In[42]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10) 
classifier.fit(x_train, y_train)


# In[43]:


knnpred = classifier.predict(x_test)
knnpred
print('-'*30)
print(x_test[:10])
print('-'*30)
print(knnpred[:10])
print('-'*30)
print(knnpred[:20])
print(y_test[:20])


# In[44]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,knnpred)
cm


# In[45]:


from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_test,knnpred))


# # KNN:95

# # random forest entroy

# In[46]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state = 0) 
classifier.fit(x_train, y_train)


# In[47]:


rdepred = classifier.predict(x_test)
rdepred
print('-'*30)
print(x_test[:10])
print('-'*30)
print(rdepred[:10])
print('-'*30)
print(rdepred[:20])
print(y_test[:20])


# In[48]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,rdepred)
cm


# In[49]:


from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_test,rdepred))


# # RDE:97

# # random forest ginni

# In[50]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='gini',random_state = 0) 
classifier.fit(x_train, y_train)


# In[51]:


rdgpred = classifier.predict(x_test)
rdgpred
print('-'*30)
print(x_test[:10])
print('-'*30)
print(rdgpred[:10])
print('-'*30)
print(rdgpred[:20])
print(y_test[:20])


# In[52]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,rdgpred)
cm


# In[53]:


from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_test,rdgpred))


# # RDE:95

# # kSVM

# In[54]:


from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state = 0) 
classifier.fit(x_train, y_train)


# In[57]:


ksvmpred = classifier.predict(x_test)
ksvmpred
print('-'*30)
print(x_test[:10])
print('-'*30)
print(ksvmpred[:10])
print('-'*30)
print(ksvmpred[:20])
print(y_test[:20])


# In[58]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,ksvmpred)
cm


# In[59]:


from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_test,ksvmpred))


# # KSVM:96

# In[60]:


from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state = 0) 
classifier.fit(x_train, y_train)


# In[62]:


svmpred = classifier.predict(x_test)
svmpred
print('-'*30)
print(x_test[:10])
print('-'*30)
print(svmpred[:10])
print('-'*30)
print(svmpred[:20])
print(y_test[:20])


# In[63]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,svmpred)
cm


# In[64]:


from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_test,svmpred))


# # SVM:97

# # Logistic

# In[66]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# In[67]:


lpred = classifier.predict(x_test)
lpred
print('-'*30)
print(x_test[:10])
print('-'*30)
print(lpred[:10])
print('-'*30)
print(lpred[:20])
print(y_test[:20])


# In[68]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,lpred)
cm


# In[69]:


from sklearn import metrics
print('accuracy:',metrics.accuracy_score(y_test,lpred))


# # logistic:95

# In[70]:


data={'Classifier':['Naive_Bayes','Decission Tree','KNN','Random Forest(Entropy)','Random Forest(Gini)','Kernel SVM','SVM','Logistic'],
     'Percentage':[91,95,95,97,95,96,97,95]}


# In[72]:


df=pd.DataFrame(data)
df


# In[74]:


breast_cancer.info()


# In[75]:


print(breast_cancer.head())


# In[ ]:





# In[77]:


rdepred = classifier.predict(sc.transform(np.array([[20.90,10.07,77.58,1299.00,0.11,0.13,0.24,0.105,0.181,0.0566,0.44,8.66,9.555,768.00,345.00,0.987,0.765,0.8765,0.987,0.0763,25.38,17.33,152.50,2089.00,0.209,0.866,0.450,0.1860,0.3613,0.118]])))
rdepred


# In[81]:


import seaborn as sns
sns.scatterplot(x='mean area',y='mean compactness',hue='target',data=breast_cancer)#data=x_test.join(y_test,how='outer'))


# In[82]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from mpl_toolkits.mplot3d import axes3d


# In[85]:


fig=plt.figure(figsize=(16,8))
ax0=fig.add_subplot(131,projection='3d')
ax1=fig.add_subplot(132,projection='3d')
ax2=fig.add_subplot(133,projection='3d')

ax0.scatter(x[:,0],x[:,1],y,color='r',label='Actual')
ax0.scatter(x[:,0],x[:,1],classifier.predict(x),color='g',label='prediction')
ax0.set_xlabel('mean radius')
ax0.set_ylabel('mean texture')
ax0.set_zlabel('target')
ax0.set_title('total data')
ax0.legend


ax1.scatter(x[:,0],x[:,1],y_train,color='r',label='Actual')
ax1.scatter(x[:,0],x[:,1],classifier.predict(x_train),color='g',label='prediction')
ax1.set_xlabel('mean radius')
ax1.set_ylabel('mean texture')
ax1.set_zlabel('target')
ax1.set_title('total data')
ax1.legend


# In[ ]:




