# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 19:29:05 2016

@author: Julian Quernheim

interesting links: 
  http://gbeced.github.io/pyalgotrade/docs/v0.18/html/ 
  https://www.backtrader.com/docu/index.html
  https://ntguardian.wordpress.com/2016/09/19/introduction-stock-market-data-python-1/
"""

import quandl
import numpy as np
import pandas as pd
from urllib.request import urlopen
import os
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
import pandas_profiling # pip install pandas-profiling

from FinanceModule.FinanceModule.quandlModule import Quandl


#print versions
print(np.version.version)



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    	data: Sequence of observations as a list or NumPy array.
    	n_in: Number of lag observations as input (X).
    	n_out: Number of observations as output (y).
    	dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    	Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    print(agg.head())
    print('rows : ' + str(len(agg)))
    print('columns : ' + str(len(agg.T)))

    return agg

def profiling(df, title, outputFile = True , outputHTMLstr = False):
    profile = df.profile_report(title=title)
    if outputFile:
        return profile.to_file(output_file="output.html")
    if outputHTMLstr:
        return profile.to_html()


def rescaleData(data):
    values = data.values
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    return scaled , scaler



def splitTrainTest(data, percentage):
    '''

    :param data:
    :param percentage:
    :return:
    '''
    values = data.values
    n_train_hours = round(len(data) * percentage)
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print('Dataset shapes')
    print('Train_X: ' + str(train_X.shape))
    print('Train_y: ' + str(train_y.shape))
    print('Test_X: ' + str(test_X.shape))
    print('Test_y: ' + str(test_y.shape))

    return train_X, train_y, test_X, test_y



def designModel(train_X):
    # design network
    model = Sequential()
    model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def fitModel(train_X, train_y, test_X, test_y):
    '''

    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    '''
    history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], shuffle=False)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    return history




def evaluateModel(yhat, actual_x, actual_y, n_input , features):
    '''

    :param yhat:
    :param actual_x:
    :param actual_y:
    :param n_input:
    :param features:
    :return:
    '''

    actual_x = actual_x.reshape((actual_x.shape[0], actual_x.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, actual_x[:, n_input*features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    actual_y = actual_y.reshape((len(actual_y), 1))
    inv_y = np.concatenate((actual_y, actual_x[:, n_input*features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # plot prediction
    pyplot.plot(inv_yhat, label='prediction for ' + codes[1])
    pyplot.plot(inv_y, label='actual ' + codes[1])
    pyplot.legend()
    pyplot.show()

    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


    return inv_y, inv_yhat , rmse


# --  MAIN  --
apiKey = "fKx3jVSpXasnsXKGnovb"
market = 'FSE'
n_input, n_output = 10, 1


# Download list of API codes
Quandl = Quandl(apiKey)
df_tickers = Quandl.getStockExchangeCodes()
# filter data API codes
l_tickers = df_tickers['code'].tolist()

# Define tickers
#l_tickers = ['FSE/AIR_X', 'EURONEXT/BOEI']  # AIR_X = Airbus  | ALV_X = Allianz SE
# Get how many features ar used
#features = len(l_tickers)

# Download the dataset
df_stockHistory = Quandl.getStockMarketData(market= market ,ListQuandleCodes=l_tickers)
pd.DataFrame.to_csv(df_stockHistory, sep=';', path_or_buf='FinanceModule/data/fse_stocks')

# Filter the dataset and keep only Close values
#df_stockHistory = df_stockHistory.rename(columns={"Last": "EURONEXT/BOEI_Close"})
df_stockHistoryCleaned = df_stockHistory.filter(like='Close' , axis=1)
# keep only rows with at least 100% filled cells
#df_stockHistoryCleaned = df_stockHistoryCleaned.dropna(axis=0, thresh=(round(len(df_stockHistoryCleaned.T)*0,05)))

df_stockHistoryCleaned = df_stockHistoryCleaned.tail(250*5)


profile_report = profiling(df_stockHistoryCleaned, 'Finance Module',outputFile = False, outputHTMLstr=True )
# Scale the dataset between 0 and 1
scaled, scaler = rescaleData(data = df_stockHistoryCleaned)
# reframe the dataset to an unsupervised learning dataset
reframed = series_to_supervised(scaled, n_input, n_output)
# split the dataset into train and test sets
train_X, train_y, test_X, test_y = splitTrainTest(data =reframed, percentage = 0.7)
# Design the model
model = designModel(train_X)
# fit network
history = fitModel(train_X, train_y, test_X, test_y)

# make a prediction
yhat = model.predict(test_X)
print('Shape yhat: ' + str(yhat.shape))

# evaluate the model
inv_y, inv_yhat , rmse= evaluateModel(yhat=yhat, actual_x=test_X, actual_y=test_y, n_input = n_input  , features = features)








# Data Exploration
'''
df_volume = data_all[['ALV_Volume','GOOGL_Volume','FB_Volume','MSCI_Volume']]
df_volume.plot()

plt.style.use('ggplot')
df = df_volume
df = df.mean()
df.plot.bar();
df.plot();


data_all.groupby(['year'])['ALV_Volume','GOOGL_Volume','FB_Volume'].mean().plot.bar();
data_all.groupby(['month'])['ALV_Volume','GOOGL_Volume','FB_Volume'].mean().plot.bar();
data_all.Series.diff().groupby(['weekday'])['ALV_Volume','GOOGL_Volume','FB_Volume'].mean().plot();



import seaborn
import matplotlib.pyplot as plt

n = 480
ts = pd.Series(np.random.randn(n), index=pd.date_range(start="2014-02-01", periods=n, freq="H"))

fig, ax = plt.subplots(figsize=(12,5))
seaborn.boxplot(ts, ts.index.dayofyear, ax=ax)



# Data Cleaning
data_all = data_all.dropna(how='any')

# M algorithmL
 
x_train=data_all.drop(['GOOGL_Volume'], axis=1)
y_train=data_all[['GOOGL_Volume']]

dates = pd.date_range(x_train.index[-1].isoformat(),periods=6)
x_test = pd.DataFrame(np.random.randn(6,17)*100,index=dates,columns=x_train.columns)



#Assumed you have, X (predictor) and Y (target) for
#training data set and x_test(predictor) of test_dataset
#Create Random Forest object
model= RandomForestClassifier()
#Train the model using the training sets and check score
model.fit(x_train, y_train)


predicted= model.predict(x_test) 

x_test['ALV_Volume_Predicted'] =predicted



importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), x_train.columns[indices])
plt.xlabel('Relative Importance')

'''