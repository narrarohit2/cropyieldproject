#!/usr/bin/env python
# coding: utf-8

# ### Importing Necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# ### Loading the Data

# In[2]:


dataset = pd.read_csv('apy.csv')
print(dataset.keys())


# ### Metadata - List of Crops

# In[3]:


description = {'Rajmash Kholar': 0, 'Other Fresh Fruits': 1, 'Rubber': 2, 'Niger seed': 3, 'Grapes': 4, 'Ber': 5, 
               'Water Melon': 6, 'Snak Guard': 7, 'Beet Root': 8, 'Ash Gourd': 9, 'Other Dry Fruit': 10, 'Onion': 11, 
               'Moth': 12, 'Pineapple': 13, 'Dry ginger': 14, 'Black pepper': 15, 'Ribed Guard': 16, 'Brinjal': 17, 
               'Citrus Fruit': 18, 'Turmeric': 19, 'other misc. pulses': 20, 'Carrot': 21, 'Khesari': 22, 'Potato': 23, 
               'Rice': 24, 'Tomato': 25, 'Other  Rabi pulses': 26, 'Pear': 27, 'Kapas': 28, 'Sweet potato': 29, 'Jowar': 30, 
               'Bajra': 31, 'Paddy': 32, 'Ricebean (nagadal)': 33, 'Castor seed': 34, 'Tea': 35, 'Plums': 36, 
               'Other Kharif pulses': 37, 'Peas & beans (Pulses)': 38, 'Arcanut (Processed)': 39, 'Pulses total': 40, 
               'Jute & mesta': 41, 'Moong(Green Gram)': 42, 'Wheat': 43, 'Beans & Mutter(Vegetable)': 44, 'Banana': 45, 
               'other oilseeds': 46, 'Perilla': 47, 'Urad': 48, 'Pome Fruit': 49, 'Lemon': 50, 'Masoor': 51, 'Korra': 52, 
               'Barley': 53, 'Rapeseed &Mustard': 54, 'Cowpea(Lobia)': 55, 'Litchi': 56, 'Sannhamp': 57, 'Lentil': 58, 
               'Maize': 59, 'Bitter Gourd': 60, 'Other Vegetables': 61, 'Arecanut': 62, 'Cucumber': 63, 'Cardamom': 64, 
               'Orange': 65, 'Apple': 66, 'Papaya': 67, 'Coffee': 68, 'Cotton(lint)': 69, 'Cashewnut Processed': 70, 
               'Pome Granet': 71, 'Sesamum': 72, 'Yam': 73, 'Peach': 74, 'Sapota': 75, 'Colocosia': 76, 'Jobster': 77, 
               'Coconut ': 78, 'Atcanut (Raw)': 79, 'Other Citrus Fruit': 80, 'Peas  (vegetable)': 81, 'Groundnut': 82, 
               'Jute': 83, 'Cabbage': 84, 'Bottle Gourd': 85, 'Gram': 86, 'Soyabean': 87, 'Redish': 88, 'Jack Fruit': 89, 
               'Small millets': 90, 'Pump Kin': 91, 'Mesta': 92, 'Mango': 93, 'Varagu': 94, 'Ginger': 95, 'Horse-gram': 96, 
               'Tapioca': 97, 'Dry chillies': 98, 'Cashewnut': 99, 'Samai': 100, 'Safflower': 101, 'Sunflower': 102, 
               'Guar seed': 103, 'Arhar/Tur': 104, 'Cond-spcs other': 105, 'Cashewnut Raw': 106, 'Turnip': 107, 
               'Lab-Lab': 108, 'Blackgram': 109, 'Garlic': 110, 'Sugarcane': 111, 'Tobacco': 112, 'Other Cereals & Millets': 113, 
               'Bean': 114, 'Ragi': 115, 'Cauliflower': 116, 'Coriander': 117, 'Linseed': 118, 'other fibres': 119, 
               'Total foodgrain': 120, 'Oilseeds total': 121, 'Drum Stick': 122, 'Bhindi': 123}


# ### Data Preprocessing

# In[4]:


def vectorize_season(num):
    one_hot = np.zeros(6,dtype=int)
    one_hot[int(num)]=1
    return one_hot
def vectorize_crops(num):
    one_hot = np.zeros(124,dtype=int)
    one_hot[int(num)]=1
    return one_hot
    
list_of_season = {"Kharif":0,"Whole Year":1,"Autumn":2,"Rabi":3,"Winter":4,"Summer":5}
crops = dataset["Crop"].values
seasons = dataset['Season'].values
data_list = []
for i in seasons:
    data_list.append(i.strip())
seasons = np.array(data_list)
#Categorical Encoding seasons
for i in list_of_season:
    seasons[seasons==i] = int(list_of_season[i])
#One-Hot Encoding seasons
encoded_seasons = []
for i in seasons:
    encoded_seasons.append(vectorize_season(i))
encoded_seasons = np.array(encoded_seasons)
#Categorical Encoding seasons
for i in description:
    crops[crops==i] = description[i]
#One-Hot Encoding seasons
encoded_crops = []
for i in seasons:
    encoded_crops.append(vectorize_crops(i))
encoded_crops = np.array(encoded_crops)
loaded_data = dataset.iloc[:,2:].values
loaded_data[:,1]=seasons
loaded_data[:,2]=crops
int_list_season=[]
for i in loaded_data[:,1]:
    int_list_season.append(int(i))
loaded_data[:,1] = int_list_season
print(loaded_data[:10])
from sklearn.model_selection import train_test_split
features = loaded_data[:,:-1]
target = loaded_data[:,-1]
updated_target=[]
for i in target:
    try:
        updated_target.append(int(i))
    except:
        updated_target.append(600)
X_train,X_test,y_train,y_test=train_test_split(features,updated_target,test_size = 0.2)
y_train = np.array(y_train)
y_test = np.array(y_test)


# ### Training Random Forest Model

# In[5]:


from sklearn.ensemble import RandomForestRegressor as rfr
reg  = rfr(100)
reg.fit(X_train,y_train)
r_prediction = reg.predict(X_test)
print(r_prediction[:10])


# ### Evaluating the RandomForrestModel

# In[6]:


from sklearn.metrics import mean_squared_error
random_for_error = mean_squared_error(y_test,r_prediction)


# ### Preprocessing for Deep learning

# In[7]:


X_train,X_test,y_train,y_test=train_test_split(features,updated_target,test_size = 0.2)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train  = y_train.reshape(-1,1)
print(y_train[:10])
y_test  = y_test.reshape(-1,1)

print(len(y_test))
print(len(X_test))


# ### Building Neural Network

# In[8]:


import tensorflow as tf
from tensorflow.keras import layers
def build_model():
  model = tf.keras.Sequential([
    layers.Flatten(),
    layers.Dense(150, activation=tf.nn.relu),
    layers.Dense(150, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()
history = model.fit(X_train, y_train, epochs=15,validation_split=0.3)


# ### Evaluation of the Deep Learning Model

# In[9]:


loss, mae, mse = model.evaluate(X_test, y_test)
print("mse",mse)
print("loss",loss)
print("mae",mae)


# ### Comparision of Deep learning model and Random Forest Regressor

# In[10]:


from matplotlib import pyplot as plt
x = ["Random_Forest","Deep_learning"]
y = [random_for_error,mse]
x2 = [6,9,11]
y2 = [6,15,7]
plt.bar(x, y, align='center')
plt.bar(x2, y2, color='g', align='center')
plt.title('Random Forrest vs Deep learning Model')
plt.xlim(-1,2)
plt.ylabel('Error')
plt.xlabel('Model')
plt.show()


# ### Fertilizer prediction and Evaluation

# In[11]:


fert_dataset = pd.read_csv('Fertilizer.csv')
X = fert_dataset.iloc[:, :-1].values
y = fert_dataset.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Fertilizer vs Production (Training set)')
plt.xlabel('Production')
plt.ylabel('Fertilizer')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Fertilizer vs Production (Test set)')
plt.xlabel('Production')
plt.ylabel('Fertilizer')
plt.show()

