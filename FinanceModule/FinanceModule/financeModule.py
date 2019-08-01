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

apiKey = "fKx3jVSpXasnsXKGnovb"

def getStockExchangeCodes(apiKey):

    # open and save the zip file onto computer
    url = urlopen('https://www.quandl.com/api/v3/databases/FSE/metadata?api_key=' + apiKey)
    output = open('zipFile.zip', 'wb')    # note the flag:  "wb"
    output.write(url.read())
    output.close()

    # read the zip file as a pandas dataframe
    df = pd.read_csv('zipFile.zip')   # pandas version 0.18.1 takes zip files
    print(df.head(10))
    # if keeping on disk the zip file is not wanted, then:
    os.remove('zipFile.zip')   # remove the copy of the zipfile on disk
    return df

def getStockMarketData(apiKey, ListQuandleCodes):
    # Downloading the data
    quandl.ApiConfig.api_key = apiKey
    n = 1
    for i in ListQuandleCodes:
        print(i)
        if n == 1:
            # initialize the dataframe
            df = quandl.get('FSE/' +i)
            df = df.rename(columns={'Open': i + '_Open', 'High': i + '_High','Low':i + '_Low','Close': i + '_Close' ,'Volume': i + '_Volume'})


        else:
            df_new = quandl.get('FSE/' +i)
            df_new = df_new.rename(columns={'Open': i + '_Open', 'High': i + '_High', 'Low': i + '_Low', 'Close': i + '_Close',
                                    'Volume': i + '_Volume'})
            df = pd.merge(df, df_new, how='outer', left_index=True, right_index=True)

        n = n + 1

    # Drop rows where any value is nan
    #df = df.dropna()

    df['year'] = df.index.year
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    return df


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
    return agg



df_tickers = getStockExchangeCodes(apiKey)
df_tickers = df_tickers.loc[df_tickers['code'] == 'AIR_X']
l_tickers = df_tickers['code'].tolist()
print(l_tickers)

df_stockHistory = getStockMarketData(apiKey, l_tickers)
print(df_stockHistory)

# keep only rows with at least 90% filled cells
#df_stockHistoryCleaned = df_stockHistory.dropna(axis=0, thresh=(round(len(df_stockHistory.T)*0.90)))
# keep only columns with at least 90% filled cells
#df_stockHistoryCleaned = df_stockHistory.dropna(axis=1, thresh=(round(len(df_stockHistory)*0.90)))

# Keep only Close values
df_stockHistoryCleaned = df_stockHistory.filter(like='Close', axis=1)



values = df_stockHistoryCleaned.values
# integer encode direction
#encoder = LabelEncoder()
#values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning



# Univariate forecasting problem (x(t-1),x)
reframed = series_to_supervised(scaled, 10, 10)
print(reframed.head())
print(len(reframed))



values = reframed.values
n_train_hours = round(len(reframed) * 0.7)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)] , shuffle=False)


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]


# plot prediction
pyplot.plot(inv_yhat, label='prediction')
pyplot.plot(inv_y, label='actual')
pyplot.legend()
pyplot.show()




# calculate RMSE
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)






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