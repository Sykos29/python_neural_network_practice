import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input


#Loading Data
dataframe = pd.read_csv('Data/Housepricedata.csv')
print(dataframe)

#input data
x = dataframe.iloc[:,0:10]

#output data
y = dataframe.iloc[:,10]

#Transforming Data
x = preprocessing.MinMaxScaler().fit_transform(x)

#Train-Test-Split (30% Validate)
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size = 0.3)


model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(10,))) 
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=32, epochs=10,validation_data=(x_val,y_val))

test = [
    [8450,7,5,856,2,1,3,8,0,548],       # 1
    [14260,8,5,1145,2,1,4,9,1,836],     # 1
    [6120,7,8,832,1,0,2,5,0,576],       # 0
    [12968,5,6,912,1,0,2,4,0,352]       # 0
    ]

test = preprocessing.MinMaxScaler().fit_transform(test)

print(model.predict(test))

model.save("ann.h5")