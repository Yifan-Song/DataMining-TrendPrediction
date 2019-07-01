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

def get_x(data, N):
    x= []
    for i in range(N, len(data)):
        x.append(data[i-N:i])
    x = np.array(x)
    return x

def get_y(data, N):
    y = []
    for i in range(N, len(data)-1):
        if(data[i]-data[i-1] > 0):
            y.append(1)
        elif(data[i]-data[i-1] < 0):
            y.append(-1)
        else:
            y.append(0)
    y = np.array(y)
    return y

class LSTMMODEL:
    test_size = 0.2                # proportion of dataset to be used as test set
    cv_size = 0.2                  # proportion of dataset to be used as cross-validation set
    N = 5                          # for feature at day t, we use lags from t-1, t-2, ..., t-N as features. 
    lstm_units=8
    dropout_prob=1                 
    optimizer='nadam'
    epochs=30
    batch_size=10
    # model_seed = 100
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
        close = np.array(df['close']).reshape(-1,1)
        last_close = close[:-1]
        last_close = np.vstack((close[0], last_close))
        change = (close - last_close)/close
        self.change_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(change)

        combined_train_cv = np.hstack((self.open_scaled[:num_train+num_cv-1],self.close_scaled[:num_train+num_cv-1],self.high_scaled[:num_train+num_cv-1],self.low_scaled[:num_train+num_cv-1]))
        combined_test_cv = np.hstack((self.open_scaled[num_train+num_cv:-1],self.close_scaled[num_train+num_cv:-1],self.high_scaled[num_train+num_cv:-1],self.low_scaled[num_train+num_cv:-1]))

        # self.x_train_cv = get_x(combined_train_cv, self.N)
        self.x_train_cv = get_x(self.change_scaled[:num_train+num_cv-1], self.N)
        self.y_train_cv = get_y(np.array(df['close'])[:num_train+num_cv],self.N).reshape(-1,1)

        #self.x_test = get_x(combined_test_cv, self.N)
        self.x_test = get_x(self.change_scaled[num_train+num_cv:-1], self.N)
        self.y_test = get_y(np.array(df['close'])[num_train+num_cv:],self.N).reshape(-1,1)
        
        print("close:\n", np.array(df['close']).reshape(-1,1)[:10])
        print("x_train:\n",get_x(change[:num_train+num_cv-1], self.N)[:10])
        print("y_train_cv:\n",'\n',self.y_train_cv[:10])
        # print("test:\n",self.x_test[:5],'\n',self.y_test[:5])

    def trainAndPred(self):
        model = Sequential()
        model.add(LSTM(units=self.lstm_units, return_sequences=True, input_shape=(self.x_train_cv.shape[1], self.x_train_cv.shape[2])))
        model.add(Dropout(self.dropout_prob))
        model.add(LSTM(units=self.lstm_units))
        model.add(Dropout(self.dropout_prob))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=self.optimizer)

        model.fit(self.x_train_cv, self.y_train_cv, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

        res = model.predict(self.x_test)
        
        print("originRes:\n",res[:5])
        res[res>=0.5] = 1
        res[res<=-0.5] = 2
        res[res<0.5] = 0
        res[res==2] = -1

        print("finalRes:\n",res[:5],'\n',self.y_test[:5])

        df = pd.DataFrame(np.hstack((res,self.y_test)))
        df.columns = ["pred_close","real_close"]

        rc = df.loc[:, "real_close"]
        pc = df.loc[:, "pred_close"]
        count = 0
        for i in range(1, df.shape[0]):
            if(rc[i] == pc[i]):
                count+=1
        print(count, rc.shape[0]-1)
        print(count/(rc.shape[0]-1))

        # rcParams['figure.figsize'] = 10, 8
        # ax = df.plot(y=['pred_close','real_close'], color=['red','blue'], grid=True)
        # plt.show()

        # df.to_csv("new_pred_data.csv")

def main():
    lstmModel = LSTMMODEL()
    lstmModel.initData("./rb000.csv")
    lstmModel.trainAndPred()

main()