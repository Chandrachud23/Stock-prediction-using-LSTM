# Stock-prediction-using-LSTM
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns #styling purpose
sns.set_style('whitegrid')
%matplotlib inline

# Predicting stock price of L&T:
#For reading stock data from yahoofinance
from pandas_datareader.data import DataReader
import pandas_datareader.data as web
#For time limits
from datetime import datetime

#Getting the stock details from web
df = web.DataReader('LT.NS','yahoo', start='1951-01-23', end=datetime.now())
#Instead of LT.NS you can use your company's ticker to find the  and you can also change the date to what you want

#Show the data
df

plt.figure(figsize=(16,8))
plt.title('Close Price History', fontsize=16)
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=16)
plt.ylabel('Close Price', fontsize=16)
plt.show()

#Creating a new dataframe with only the Close column
data = df.filter(['Close'])#function is used to Subset rows or columns of dataframe according to labels in the specified index.

#Converting the dataframe to a numpy array
dataset = data.values
print(len(dataset))#output1

#Getting the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .8 ))# Ceil() is basically used to round up the values specified in it. It rounds up the value to the nearest greater integer.

training_data_len#output2

# Scaling the data
from sklearn.preprocessing import MinMaxScaler
#LSTM are sensitive to the scale of the data. so we apply MinMax scaler

scaler = MinMaxScaler(feature_range=(0,1))#scaling features within (0,1)
scaled_data = scaler.fit_transform(dataset)
scaled_data

# Creating the training data set

#Creating the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]#you need to select rows starting from zero till training_data_len and all the columns from your scaled_data.
#Train Dataset: Used to fit the machine learning model.

#Splitting the data into x_train and y_train data sets
x_train = []
y_train = []

#Creating a training data set that contains the past 60 day closing price values that we want to use to predict the 61st closing price value.
#So the first column in the ‘x_train’ data set will contain values from the data set from index 0 to index 59 (60 values total)and the second column will contain values from the data set from index 1 to index 60 (60 values) and so on and so forth.
#The ‘y_train’ data set will contain the 61st value located at index 60 for it’s first column and the 62nd value located at index 61 of the data set for it’s second value and so on and so forth.

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Converting the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#So basically, we're showing the the model [from 60 to last], in order, and having it make the prediction. (60 sequences of 1 element)
x_train.shape

# Building the LSTM Model
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.
from tensorflow import keras

from keras.models import Sequential #to initialize NN as a sequnce of layers
from keras.layers import Dense, LSTM  #to add fully connected layers
from keras.layers import Dropout #Dropout is a technique used to prevent a model from overfitting.

#Building the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))#input layer is 3D
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences= False))
#no. of memory units-50
#input_shape=(1, 1) means the 1st element is the time step and the 2nd element is no. of features
#This return_sequences flag is used for when you're continuing on to another recurrent layer. If you are, then you want to return sequences. If you're not going to another recurrent-type of layer, then you don't set this to true.

from tensorflow.keras import layers #basic building blocks of neural networks in Keras.
from tensorflow.keras import activations
model.add(layers.Dense(25, activation=activations.relu))
model.add(Dense(1))
#Built the LSTM model to have two LSTM layers with 50 neurons and two Dense layers, one with 25 neurons and the other with 1 neuron.

# Compile the model
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#The learning rate(lr) controls how much to change the model in response to the estimated error each time the model weights are updated. 
model.compile(optimizer=opt, loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=2)#Verbose=2 will just mention the number of epoch like this: 1/100...

To understand clearly about how LSTM models work in brief go through the git repository. https://github.com/MohammadFneish7/Keras_LSTM_Diagram
# Creating the testing data set
#Creating a new array containing scaled values 
test_data = scaled_data[training_data_len - 60: , :]
#Test Dataset: Used to evaluate the fit machine learning model.
#Setting the time step as 60 (as seen previously)

#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predicted_data

# Visualizing the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='upper left')
plt.show()
#Convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data to be 3-D in the form [number of samples, number of time steps, and number of features]. This needs to be done, because the LSTM model is expecting a 3-dimensional data set.
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

#Predicted price values 
predicted_data = model.predict(x_test)
predicted_data = scaler.inverse_transform(predicted_data) #unscaling

# Plotting and Visualizing the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predicted_data

# Visualizing the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='upper left')
plt.show()

# Show the valid and predicted prices
print(valid.tail())

# Visualizing the comparison for testing
plt.figure(figsize=(16,8))
plt.plot(y_test,color='red',label='Real Stock price')
plt.plot(predicted_data,color='blue',label='Predicted Stock price')

plt.title('Stock price prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')

plt.legend()
plt.show()

# Calculating Root mean square error
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test, predicted_data))
print(rmse)

print('RMSE in terms of % of the orignal value is', round((rmse/y_test.mean()*100), 2) , '%')
#we take the avg because it would be a true representative of the real stock values

# Calculating Mean Absolute error
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test,predicted_data)
print(mae)
print('MAE in terms of % of the orignal value is', round((mae/y_test.mean()*100), 2) , '%')
