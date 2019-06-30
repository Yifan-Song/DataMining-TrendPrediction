import math
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import time

from datetime import date
from matplotlib import pyplot as plt
from numpy.random import seed
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import plot_model

def get_x(data, N, offset):
    x= []
    for i in range(offset, len(data)):
        x.append(data[i-N:i])
    x = np.array(x)
    return x

def get_y(data, N):
    y = []
    for i in range(N, len(data)):
        if(data[i+1]-data[i] > 0):
            y.append(1)
        elif(data[i+1]-data[i] < 0):
            y.append(-1)
        else:
            y.append(0)
    y = np.array(y)
    return y

class LSTMMODEL:
    test_size = 0.2                # proportion of dataset to be used as test set
    cv_size = 0.2                  # proportion of dataset to be used as cross-validation set
    N = 20                          # for feature at day t, we use lags from t-1, t-2, ..., t-N as features. 
    lstm_units=128                  # lstm param. initial value before tuning.
    dropout_prob=1                 # lstm param. initial value before tuning.
    optimizer='nadam'               # lstm param. initial value before tuning.
    epochs=5                       # lstm param. initial value before tuning.
    batch_size=10                   # lstm param. initial value before tuning.
    # model_seed = 100
    paramNum = 4
    close_scaler = MinMaxScaler(feature_range=(0, 1))

    def __init__(self):
        self.data_path = ""

    def initData(self,path):
        self.data_path = path
        df = pd.read_csv(self.data_path, sep = ",")

        num_cv = int(self.cv_size*len(df))
        num_test = int(self.test_size*len(df))
        num_train = len(df) - num_cv - num_test

        self.open_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(df['open']).reshape(-1,1))
        self.close_scaled = self.close_scaler.fit_transform(np.array(df['close']).reshape(-1,1))
        self.high_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(df['high']).reshape(-1,1))
        self.low_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(df['low']).reshape(-1,1))
        # self.amount_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(df['amount']).reshape(-1,1))

        # cv = df[num_train:num_train+num_cv][['date', 'close']]
        # train_cv = df[:num_train+num_cv][['date', 'close']]
        # test = df[num_train+num_cv:][['date', 'close']]
        combined_train_cv = np.hstack((self.open_scaled[:num_train+num_cv],self.close_scaled[:num_train+num_cv],self.high_scaled[:num_train+num_cv],self.low_scaled[:num_train+num_cv]))
        combined_test_cv = np.hstack((self.open_scaled[num_train+num_cv:],self.close_scaled[num_train+num_cv:],self.high_scaled[num_train+num_cv:],self.low_scaled[num_train+num_cv:]))

        # self.x_train, self.y_train = get_x_y(self.close_scaled[:num_train], self.N, self.N)
        # self.x_cv, self.y_cv = get_x_y(self.close_scaled[num_train:num_train+num_cv], self.N, self.N)

        self.x_train_cv = get_x(combined_train_cv, self.N, self.N)
        # self.x_train_cv = get_x(self.close_scaled[:num_train+num_cv], self.N, self.N)
        self.y_train_cv = self.close_scaled[self.N:num_train+num_cv]
        self.x_test = get_x(combined_test_cv, self.N, self.N)
        # self.x_test = get_x(self.close_scaled[num_train+num_cv:], self.N, self.N)
        self.y_test = self.close_scaled[self.N+num_train+num_cv:]
        # print(self.x_test[:5],'\n',self.y_test[:5])

    def trainAndPred(self):
        model = Sequential()
        model.add(LSTM(units=self.lstm_units, return_sequences=True, input_shape=(self.x_train_cv.shape[1], self.paramNum)))
        model.add(Dropout(self.dropout_prob))
        model.add(LSTM(units=self.lstm_units))
        model.add(Dropout(self.dropout_prob))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=self.optimizer)

        model.fit(self.x_train_cv, self.y_train_cv, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

        res = model.predict(self.x_test)
        res_inv = self.close_scaler.inverse_transform(res)
        y_test_inv = self.close_scaler.inverse_transform(self.y_test)

        print(res_inv[:5],'\n',y_test_inv[:5])

        df = pd.DataFrame(np.hstack((res_inv,y_test_inv)))
        df.columns = ["pred_close","real_close"]

        rc = df.loc[:, "real_close"]
        pc = df.loc[:, "pred_close"]
        count = 0
        for i in range(1, df.shape[0]):
            if((pc[i]-pc[i-1])*(rc[i]-rc[i-1]) > 0):
                count+=1
        print(count, rc.shape[0]-1)
        print(count/(rc.shape[0]-1))

        rcParams['figure.figsize'] = 10, 8
        ax = df.plot(y=['pred_close','real_close'], color=['red','blue'], grid=True)
        plt.show()

        # df.to_csv("new_pred_data.csv")
        print(df.head())

def main():
    lstmModel = LSTMMODEL()
    lstmModel.initData("./rb000.csv")
    lstmModel.trainAndPred()

main()