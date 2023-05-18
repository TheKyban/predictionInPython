import tensorflow as tf
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas
import numpy as np


# variables
set = 500 # train model with 500 days
day = 30 # predict days
time_step = 100 # to predict next day so you have to give previous days 

hidden_layer = 50
dense_layer = 1
epochs = 30
batch_size = 64
verbose = 1

all_r2 = []
all_msError = []


######################### Functions #############################

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

#######################################################################################


#     READ DATA
#####################

data = pandas.read_csv('AAPL.csv')["Close"]


#     NORMALIZING
######################

scaler = MinMaxScaler(feature_range=(0, 1))
close_values = scaler.fit_transform(np.array(data).reshape(data.shape[0], 1))


#     BUILDING RNN MODEL
#############################

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_layer,return_sequences=True,input_shape=(time_step,1)),
    tf.keras.layers.LSTM(hidden_layer,return_sequences=True),
    tf.keras.layers.LSTM(hidden_layer),
    tf.keras.layers.Dense(dense_layer)
])

#     COMPILING THE MODEL
################################

model.compile(optimizer='adam',loss='mean_squared_error')




steps = cal_step(len(close_values),set,day) # total steps
print(steps)


#        MAIN FUNCTION 
# The model is train every time when the loop start so to completion of program take so much time
# If we train the model once with all data then result may good as well as time may take less than before.(not implemented) 
################################

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




"""
WHEN EPOCH == 30

all r2:  [0.35439323641102616, -0.922684304267636, 0.5928271620109772, 0.712399526432967, -0.34980721032579876, 0.4561150262172956, 0.7279265593766792, 0.833289614696494, 0.8763943876096212, 0.9082982805858164, 0.684432074638593, 0.9066883551015185, 0.9207169516379752, 0.9144338410872812, 0.9289161978448965, 0.9202430353593197, 0.9164018074131351, 0.5339467001412996, 0.9010056227119667, 0.9217150057916944, 0.8952566836342135, 0.7640329279258262, 0.87634701374337, 0.6165163110062064, 0.7048409822973969, 0.8765168046879412, 0.8790599108422616, 0.8275348800950189, 0.897744381135409, 0.9199565227605166, 0.6836636505357128, 0.8339009126910275, 0.8830389869229711, 0.909555808466432, 0.9058218892377944, 0.6426836874532915, 0.8946060613106336, 0.032152823816138154, 0.9197578352999789, -0.09361746076765898, 0.8471230428182602, 0.8947756388591762, 0.9192875085399403, -0.687939864947837, 0.9198623877902276, 0.8183852367487783, 0.9012553031839262, -2.8345915772296606, 0.5352285250157951, 0.9208984202991138, 0.9123391755302908, 0.7951952016702146, 0.8464099113434858, 0.1525866153323019, 0.7211691262587577, 0.5550340857639354, 0.8995239606674515, 0.7574066940325137, 0.7630106871213527, 0.7653604898666059, 0.22690049902881926, 0.8197110267319334, 0.08462599144610428, 0.6284686895592402, -2.4101079330974695, 0.024569795340179335, 0.6806456714575145, -1.139993762095255, -7.125026350429476, -8.926488010604348, -0.9706297836748066, -0.3994140876142689, 0.4480477183499143, -6.838507240847812, -13.937636055178167, -15.702303606814155, -17.81228592909155, -14.036455332603715, -26.02421683924311, -26.049845778407608, -25.917657981969374, -73.43721341261516, -56.26510572102876, -67.44286128211733, -93.23027026989546, -92.35839747604048, -122.18266294033845, -159.07974969698145, -185.65580858017788, -174.0518944386169, -155.01886285692123, -123.96106642215199, -142.51344778643775, -124.1075334350437, -125.99765982798083]


all mean square Error:  [0.35231567493938576, 1.0492328404178208, 0.2221993035444986, 0.1569471707427241, 0.7366066546458113, 0.29680482367071076, 0.14847387492258807, 0.09097593958139398, 0.06745312659601788, 0.05004277369850973, 0.17220935851859873, 0.05092133014430875, 0.043265749788058416, 0.04669452169571438, 0.03879131871901713, 0.04352437182084365, 0.045620577890514956, 0.25433110704209205, 0.05402246819031518, 0.04272109917002754, 0.057159736053037764, 0.12877017861326365, 0.06747897910656155, 0.2092718390449052, 0.16107196268864296, 0.0673863220673096, 0.06599851727384505, 0.09411637013380146, 0.05580216845367361, 0.04368072532473942, 0.1726286971496579, 0.09064234663026699, 0.06382708575535051, 0.04935652502671568, 0.05139417138886584, 0.19499197487015937, 0.0575147887817118, 0.5281662930850056, 0.043789151553690606, 0.5968007083351371, 0.08342686506682083, 0.05742224819533808, 0.04404581474133116, 0.9211298677700986, 0.04373209595383324, 0.09910944479284839, 0.05388621443809515, 2.0925845202400306, 0.2536315991972526, 0.04316672007301334, 0.04783760685390081, 0.11176453659385961, 0.0838160298200818, 0.4624440687581551, 0.1521614905797777, 0.24282345731264166, 0.05483102967493081, 0.13238619721482048, 0.1293280281887182, 0.12804571148006708, 0.4218900541961666, 0.09838594463906045, 0.4995310300853, 0.20274927677552337, 1.8609384935599638, 0.5323044464412999, 0.17427564602915463, 1.167821325890988, 4.433928366299328, 5.417008495685965, 1.0753972874005726, 0.7636777472043298, 0.3012072543398692, 4.27757130939036, 8.15165457611213, 9.114655700886722, 10.266099409206678, 8.205581490120375, 14.747452679240888, 14.761438711465095, 14.689302179837092, 40.621317128531565, 31.250283470425682, 37.35012429755739, 51.42263548371839, 50.946843609852785, 67.22231779821676, 87.35751890995705, 101.86040619479856, 95.5280053073049, 85.14144223821519, 68.19281479165997, 78.31708103347486, 68.27274367006335, 69.30420925159859]

"""