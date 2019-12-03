import pandas as pd
import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
from datetime import datetime


import keras
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import AveragePooling1D

from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Add
from keras.layers import Concatenate
from keras.layers import GaussianNoise

from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.utils.vis_utils import plot_model



def dataloader(stock_name):
    my_share = share.Share(stock_name)
    symbol_data = None

    try:
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_YEAR,30,
                                            share.FREQUENCY_TYPE_DAY,1)
        print(stock_name+" has been downloaded")

    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)
        
    df = pd.DataFrame(symbol_data)

    df['timestamp'] = [datetime.fromtimestamp(t/1000) for t in df['timestamp']]

    # test = df[(df['timestamp'] >= '2016-1-1 01:00:00') & (df['timestamp'] <= '2018-12-31 04:00:00')]
    return df[(df['timestamp'] >= '1995-01-01 01:00:00') & (df['timestamp'] <= '2019-12-31 04:00:00')]

import numpy as np
import math

def get_data(stock_name):
    train = []
    test = []
    
    for name in stock_name:
        df = dataloader(name)
        train.append( list(df[(df['timestamp'] >= '1995-01-01 01:00:00') & (df['timestamp'] <= '2015-12-31 04:00:00')]['close']) )
        test.append( list(df[(df['timestamp'] >= '2016-01-01 01:00:00') & (df['timestamp'] <= '2017-12-31 12:00:00')]['close']) )

    return train, test



def Res_model():
  
    x_input = Input(shape=(20,1,))

    x = GaussianNoise(stddev=10)(x_input)
    y = Conv1D(filters=20, kernel_size=3, activation='tanh')(x)
    y = Conv1D(filters=20, kernel_size=3, activation='relu')(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Dropout(0.2)(y)

    x = Conv1D(filters=20, kernel_size=3, activation=None)(x)
    x = Conv1D(filters=20, kernel_size=3, activation=None)(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Add()([x,y])

    y = Conv1D(filters=20, kernel_size=3, activation='tanh')(x)
    y = Conv1D(filters=20, kernel_size=3, activation='relu')(y)
    y = Conv1D(filters=20, kernel_size=3, activation='relu')(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Dropout(0.2)(y)

    x = Conv1D(filters=20, kernel_size=3, activation=None)(x)
    x = Conv1D(filters=20, kernel_size=3, activation=None)(x)
    x = Conv1D(filters=20, kernel_size=3, activation=None)(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Add()([x,y])

    x = Flatten()(x)
    x = Dense(units=1)(x)


    Res1D = Model(input=x_input, output=x)
    Res1D.compile(loss= 'logcosh', optimizer=Adam(), metrics=['MSE'])

    return(Res1D)

  
t_data, v_data = get_data(["AAPL"])
t_data, v_data = t_data[0], v_data[0]

train_data = []
res_train = []
valid_data = []
res_valid = []

  
for i in range(len(t_data) - 21):
    data = []
    for j in range(20):
        data.append([ t_data[i+j] ])
    train_data.append(np.array(data))
    res_train.append(t_data[i+20])

for i in range(len(v_data) - 21):
    data = []
    for j in range(20):
        data.append([ v_data[i+j] ])
    valid_data.append(np.array(data))
    res_valid.append(v_data[i+20])


model = Res_model()

#Time to train
model.fit(
    x = np.array(train_data),
    y = np.array(res_train),
    verbose=1,
    validation_data=(np.array(valid_data), res_valid),
    epochs=15,
    batch_size=30
)

model.save("./models/TS_AAPL")

