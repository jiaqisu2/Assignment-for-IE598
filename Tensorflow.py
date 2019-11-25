#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("/Users/sujiaqi/Desktop/IE580/creditcard.csv")
df = df.dropna(how = 'any', axis = 0)#drop row with na data
df.head()


# In[2]:


#General description of data
print(df.describe())


# In[3]:


#check numbers of two classes
print ("Fraud Transaction Number:")
print (df.Class[df.Class == 1].count())
print ("Normal Transaction Number:")
print (df.Class[df.Class == 0].count())


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#heat map
cm = np.corrcoef(df.values.T)
sns.set(font_scale=1.5)
fig, ax = plt.subplots(figsize=(18,18)) 
sns.heatmap(cm,
            cbar=True,
            annot=True,
            square=True,
            fmt='.2f',
            annot_kws={'size': 5},
            yticklabels=df.columns,
            xticklabels=df.columns, ax=ax)
plt.savefig('heatmap.png', dpi=300)
plt.show()


# In[94]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
#preprocessing
X = df.iloc[:, :-1].values
y = df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1)

#Standardize
ss= StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

sm = SMOTE(1,random_state=1)
X_resample,y_resample = sm.fit_resample(X_train, y_train)


# In[95]:


from tensorflow.keras.utils import to_categorical
y_resample = to_categorical(y_resample)
y_test = to_categorical(y_test)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model_relu = Sequential()
model_relu.add(Dense(50,input_dim=X_resample.shape[1],activation='relu'))
model_relu.add(Dense(32,activation='relu'))
model_relu.add(Dense(2,activation='softmax'))


# In[96]:


model_relu.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',metrics = ['accuracy'])
model_training = model_relu.fit(X_resample, y_resample, epochs = 10, validation_split = .2)


# In[98]:


plt.plot(model_training.history['val_loss'], 'r', model_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


# In[101]:


#add callback
from tensorflow.keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)
model_relu.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',metrics = ['accuracy'])
model_training = model_relu.fit(X_resample, y_resample,epochs = 10, validation_split = .2,callbacks = [early_stopping_monitor])


# In[102]:


plt.plot(model_training.history['val_loss'], 'r', model_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


# In[103]:


#use optimazer adam
model_relu.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
model_training = model_relu.fit(X_resample, y_resample,epochs = 10, validation_split = .2,callbacks = [early_stopping_monitor])


# In[104]:


plt.plot(model_training.history['val_loss'], 'r', model_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


# In[105]:


model_hsig = Sequential()
model_hsig.add(Dense(50,input_dim=X_resample.shape[1],activation='hard_sigmoid'))
model_hsig.add(Dense(32,activation='hard_sigmoid'))
model_hsig.add(Dense(2,activation='softmax'))


# In[108]:


#use optimazer sgd
model_hsig.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',metrics = ['accuracy'])
model_training = model_hsig.fit(X_resample, y_resample,epochs = 10, validation_split = .2,callbacks = [early_stopping_monitor])


# In[109]:


plt.plot(model_training.history['val_loss'], 'r', model_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


# In[110]:


#use optimazer adam
model_hsig.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
model_training = model_hsig.fit(X_resample, y_resample,epochs = 10, validation_split = .2,callbacks = [early_stopping_monitor])


# In[111]:


plt.plot(model_training.history['val_loss'], 'r', model_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


# In[ ]:




