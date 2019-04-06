
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import sys
import os
import time


# In[2]:


x_train_path = sys.argv[1]
train_path = os.path.abspath(x_train_path)
path_train = os.path.dirname(train_path)
os.chdir(path_train)

x_valid_path = sys.argv[2]
valid_path = os.path.abspath(x_valid_path)
path_valid = os.path.dirname(valid_path)
os.chdir(path_valid)

x_test_path = sys.argv[3]
test_path = os.path.abspath(x_test_path)
path_test = os.path.dirname(test_path)
os.chdir(path_test)

x_train = pd.read_csv(x_train_path, header=0, low_memory = False)
x_valid = pd.read_csv(x_valid_path, header=0, low_memory = False)
x_test = pd.read_csv(x_test_path, header=0, low_memory = False)

plot = sys.argv[5]

print(x_test)

y_train = x_train[" Rich?"]
y_valid = x_valid[" Rich?"]

x_train = x_train.drop([" Rich?"], axis=1)
x_valid = x_valid.drop([" Rich?"], axis=1)
x_test = x_test.drop([" Rich?"], axis=1)

# In[3]:


for i in range(x_train.shape[1]):
    print(str(i) + " - " + str(type(x_train.iloc[0,i])))


# In[4]:


header = list(x_train)
# print(header)


# In[5]:


# print(x_train_new.shape)
# print('---------------')
# print(x_train_new.head())


# In[6]:


# for i in range(x_train_new.shape[1]):
#     print(str(i)+'-'+str(x_train_new[header[i]].unique()))


# In[7]:

print("Merging")

x_train['check'] = 0
x_valid['check'] = 1
x_test['check'] = 2

merged = pd.concat([x_train, x_test, x_valid])


# In[8]:

print("Changing")

string = []

for i in range(1,merged.shape[1]):
    if(type(merged.iloc[0,i])==str):
        string.append(i)


# In[9]:

print("Chaning all")

lb_make = LabelEncoder()

for i in range(len(string)):
    merged[header[string[i]]+'_code'] = lb_make.fit_transform(merged[header[string[i]]])


# In[10]:

print("Removing")

for i in range(len(string)):
    merged = merged.drop(header[string[i]], axis=1)


# In[11]:


train = merged[merged['check']==0]
valid = merged[merged['check']==1]
test = merged[merged['check']==2]

train = train.drop(['check'], axis=1)
valid = valid.drop(['check'], axis=1)
test = test.drop(['check'], axis=1)

print("Separated")

# In[12]:


x_train = np.array(train)
x_valid = np.array(valid)
x_test = np.array(test)

y_train = np.array(y_train)
y_valid = np.array(y_valid)

# y_train = x_train[0]
# y_valid = x_valid[0]

# x_train = x_train[:,1:]
# x_test = x_test[:,1:]
# x_valid = x_valid[:,1:]

# In[13]:

print("Starting making tree")

clf_entropy = DecisionTreeClassifier(criterion="entropy",splitter='best',random_state=0,max_depth=None)
clf_entropy.fit(x_train, y_train)
clf_entropy.predict(x_test)

b = np.arange(2, 1000, 10)

y = []
z = []
for i in range(len(b)):
    clf_entropy = DecisionTreeClassifier(criterion="entropy",splitter='best',random_state=0,max_depth=None,max_leaf_nodes=b[i])
    clf_entropy.fit(x_train, y_train)
    y.append(accuracy_score(y_train,clf_entropy.predict(x_train)))
    z.append(accuracy_score(y_valid,clf_entropy.predict(x_valid)))

print("Giving output")

x_output = sys.argv[4]
actual_path_out = os.path.abspath(x_output)
path_out = os.path.dirname(actual_path_out)
os.chdir(path_out)
np.savetxt(x_output, clf_entropy.predict(x_test))

print("Plotting")

plt.plot(b, y, color='r')
plt.plot(b, z, color='b')
plt.title('Part A')
plt.xlabel('No. of nodes')
plt.ylabel('Accuracy')
plt.legend(['Training dataset', 'Validation dataset'])
plt.savefig(plot)

