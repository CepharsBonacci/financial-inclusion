#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[111]:


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
var=pd.read_csv("VariableDefinitions.csv")


# In[112]:


train.info()


# In[113]:


train.shape


# In[114]:


train.size


# In[115]:


train.columns


# In[116]:


new_train=train.drop('uniqueid',axis=1)
train.update(new_train)


# In[117]:


new_train


# In[118]:


train.describe()


# In[119]:


train.isnull().sum()


# In[120]:


sns.displot(train)


# In[121]:


sns.histplot(train)


# In[122]:


train.corr()


# In[123]:


sns.heatmap(train.corr(), annot=True, cmap='coolwarm', center=0)
plt.show()


# In[124]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[ ]:





# In[125]:


train['bank_account']=le.fit_transform(train['bank_account'])
X=train.drop(['bank_account','uniqueid'],axis=1)
y=train['bank_account']


# In[126]:


X


# In[127]:


train.columns


# In[128]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[129]:


train.columns


# In[130]:


cat_var= ['country','location_type','cellphone_access','gender_of_respondent','relationship_with_head','marital_status','education_level']
num_var=['household_size','age_of_respondent']
target='bank_account'


# In[131]:


x_cat=pd.get_dummies(train[cat_var],drop_first=True)
y_cat=train[num_var]
X=pd.concat([x_cat,y_cat],axis=1)
y=train[target]


# In[132]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[133]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
rf.fit(X_train,y_train)


# In[134]:


y_pred=rf.predict(X_test)
y_pred


# In[135]:


test_X_cat = pd.get_dummies(test[cat_var], drop_first=True)
test_X_num = test[num_var]
test_X = pd.concat([test_X_cat, test_X_num], axis=1)


# In[136]:


test['bank_account'] = rf.predict(test_X)


# In[137]:


from sklearn.metrics import accuracy_score, f1_score
accuracy=accuracy_score(y_pred,y_test)
print("Acuraccy",accuracy)
f1score=f1_score(y_pred,y_test)
print("F1score",f1score)


# In[138]:


train = train.dropna(subset=['uniqueid'], axis=0, inplace=False)
missing_ids = ['uniqueid_7867 x Kenya', 'uniqueid_6722 x Kenya', 'uniqueid_6714 x Kenya', 'uniqueid_8103 x Kenya', 'uniqueid_8657 x Kenya']
train = train[~train['uniqueid'].isin(missing_ids)]


# In[139]:


test['uniqueid'] = test['uniqueid'] + ' x ' + test['country']
test['bank_account'].fillna(0, inplace=True)  # fill missing values with 0
output_df = test[['uniqueid', 'bank_account']]
output_df.columns = ['uniqueid', 'bank_account']  # add column headers
output_df.set_index('uniqueid', inplace=True)
output_df.reset_index(drop=False, inplace=True)  # reset index and keep old index as a column
output_df.index.name = None  # remove the name of the index
print(output_df)


# In[142]:


from IPython.display import FileLink

output_df.to_csv('mark x.csv', index=False)
FileLink('mark x.csv')


# In[ ]:




