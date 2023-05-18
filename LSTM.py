import tensorflow as tf
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas
import numpy as np


# variables
set = 500
day = 30
time_step = 100

hidden_layer = 50
dense_layer = 1
epochs = 30
batch_size = 64
verbose = 1

all_r2 = []
all_msError = []


# Functions 

def create_train_set(data, time_step): # for training
    x_data, y_data = [], []

    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        x_data.append(a)
        y_data.append(data[i+time_step, 0])

    return np.array(x_data), np.array(y_data)

def create_test_set(all_data, set_data, time_step, days): # for test
    x_test, y_test = [], []
    for i in range(days):
        x = all_data[set_data.shape[0] - time_step + i: set_data.shape[0] + i] # 450 - 500
        y = all_data[set_data.shape[0]+i] # 501
        x_test.append(x)
        y_test.append(y)

    return np.array(x_test),np.array(y_test)


def cal_step(total_set, set, days): # Calculate steps
    return int((total_set-set)/days)


# Read Doc
data = pandas.read_csv('AAPL.csv')["Close"]


# Normalizing
scaler = MinMaxScaler(feature_range=(0, 1))
close_values = scaler.fit_transform(np.array(data).reshape(data.shape[0], 1))


# Building the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_layer,return_sequences=True,input_shape=(time_step,1)),
    tf.keras.layers.LSTM(hidden_layer,return_sequences=True),
    tf.keras.layers.LSTM(hidden_layer),
    tf.keras.layers.Dense(dense_layer)
])

# Compile the model
model.compile(optimizer='adam',loss='mean_squared_error')

steps = cal_step(len(close_values),set,day) # total steps
print(steps)


def Main():

    iterated = 0

    for i in range(steps):
        last_index = set + iterated  # 500 + 0

        close_value = close_values[iterated:last_index]  # 0-500

        x_train ,y_train = create_train_set(close_value,time_step)
        x_test , y_test = create_test_set(close_values,close_value,time_step,day)

        iterated += day  # 30

        # reshaping
        x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
        y_train = y_train.reshape(y_train.shape[0],1)

        x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
        y_test = y_test.reshape(y_test.shape[0],1)

        # print(x_train.shape,y_train.shape)
        # print(x_test.shape,y_test.shape)

        # Training the model
        model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=verbose)

        # Prediction
        y_predict = model.predict(x_test)

        # inverse normalizing
        y_predict = scaler.inverse_transform(y_predict)
        y_test = scaler.inverse_transform(y_test)

        r2 = r2_score(y_test,y_predict) # r2
        mean_err = mean_squared_error(y_test,y_predict) # mean square error

        all_r2.append(r2)
        all_msError.append(mean_err)
        # print(y_predict,y_test)

        print(r2)
        print(mean_err)

Main()
print("all r2: ", all_r2)
print("all Error: ", all_msError)