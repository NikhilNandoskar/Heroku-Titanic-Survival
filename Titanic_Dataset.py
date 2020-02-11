
# coding: utf-8

# In[1]:


#Importing Libraries
import numpy as np
import pandas as pd
import pickle
from LR_Regularization_Dropout_Adam import *
from LR_Regularization_Dropout_Adam import L_layer_model

# In[2]:


#Importing Data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#Data Preprocessing
#dataset = pd.concat([dataset_train, dataset_test], sort = False)
#Drroping Unwanted Data
df_train = df_train.drop(['Ticket', 'PassengerId','Cabin','Fare'], axis = 1 )
df_test = df_test.drop(['Ticket','Cabin','Fare'], axis = 1 )


# In[3]:


#print("Training DataFrame",df_train.columns)
#print("Testing DataFrame",df_test.columns)


# In[4]:


#Handling NaN Values
df_train['Embarked'] = df_train['Embarked'].fillna(value=df_train['Embarked'].value_counts().idxmax())
df_train = df_train.drop(['Name'],axis=1)
df_train['Age'] = df_train['Age'].fillna(value=df_train["Age"].mean())
#print("Training Dataframe", df_train.columns)

#Handling NaN Values
df_test['Embarked'] = df_test['Embarked'].fillna(value=df_test['Embarked'].value_counts().idxmax())
df_test = df_test.drop(['Name'],axis=1)
df_test['Age'] = df_test['Age'].fillna(value=df_test['Age'].mean())
#print("Testing Dataframe", df_test.columns)
#print(df_test)


# In[5]:


X = df_train.iloc[:, 1:].values
y = df_train.iloc[:, 0].values
#print(np.unique(y).shape[0])

X_test = df_test.iloc[:,1:].values
passenger_id = df_test.iloc[:, 0].values.astype(int)


# In[6]:

"""
from sklearn.preprocessing import Imputer
imp_Age = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imp_Age.fit(X[:, 2:3])
X[:, 2:3] = imp_Age.transform(X[:, 2:3])

# Testing Data
imp_Age_test = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imp_Age_test.fit(X_test[:, 2:3])
X_test[:, 2:3] = imp_Age_test.transform(X_test[:, 2:3])"""


# In[7]:


#Categorical Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_sex = LabelEncoder()
X[:, 1] = labelencoder_sex.fit_transform(X[:, 1]).astype(float)  #Male == 1, Female == 0

labelencoder_e = LabelEncoder()
X[:, -1] = labelencoder_e.fit_transform(X[:, -1]).astype(float)   #S = 2, C = 0, Q = 1
#Converting numeric values of Embarked to Binary Classifications
#onehotencoder = OneHotEncoder(categorical_features = [-1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]

# Testing
X_test[:, 1] = labelencoder_sex.fit_transform(X_test[:, 1]).astype(float)   #Male == 1, Female == 0

X_test[:, -1] = labelencoder_e.fit_transform(X_test[:, -1]).astype(float)   #S = 2, C = 0, Q = 1
#Converting numeric values of Embarked to Binary Classifications
#onehotencoder = OneHotEncoder(categorical_features = [-1])
#X_test = onehotencoder.fit_transform(X_test).toarray()
#X_test = X_test[:, 1:]


# In[8]:


"""#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_val, y_val, y_val = train_test_split(X, y, test_size = 0.319, random_state = 0, shuffle = False)"""


# In[9]:


#print(X_test.shape)
#print(y.shape)


# In[10]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)


# In[11]:

Output_classes = np.unique(y).shape[0]




# In[19]:

    
#model = Logistic_Regression(X, y,Output_classes, layers_dims, predict_result,
                    #activation_type, reg_type,keep_prob, mini_batch_size,
                    #n, learning_rate,lambd, num_epochs)

l_params = L_layer_model(X, y,Output_classes, layers_dims=[X.shape[1],10,Output_classes-1], 
                       predict_result=False,activation_type="binary", 
                       reg_type="l2",keep_prob=0.8, mini_batch_size=64, n=1, 
                       learning_rate = 0.002,lambd=0.01, num_epochs =500)




# In[13]:

model_to_pickle = "Logisitic_Regression.pkl"
with open(model_to_pickle, 'wb') as file:
    pickle.dump(l_params, file)






