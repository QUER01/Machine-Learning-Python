# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 19:29:05 2016

@author: Julian Quernheim

interesting links: 
  http://gbeced.github.io/pyalgotrade/docs/v0.18/html/ 
  https://www.backtrader.com/docu/index.html
  https://ntguardian.wordpress.com/2016/09/19/introduction-stock-market-data-python-1/
"""

from sklearn.ensemble import RandomForestClassifier
import quandl
import numpy as np
import pandas as pd
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
import os

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

    for i in ListQuandleCodes:
        n = 1
        print(i)
        if n == 1:
            # initialize the dataframe
            df = quandl.get('FSE/' +i)
            df = df.rename(columns={'Open': i + '_Open', 'High': i + '_High','Low':i + '_Low','Close': i + '_Close' ,'Volume': i + '_Volume'})
            n= n+ 1

        else:
            df_new = quandl.get('FSE/' +i)
            df_new = df_new.rename(columns={'Open': i + '_Open', 'High': i + '_High', 'Low': i + '_Low', 'Close': i + '_Close',
                                    'Volume': i + '_Volume'})
            df = pd.merge(df, df_new, how='outer', left_index=True, right_index=True)
            n = n + 1

    df['year'] = df.index.year
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    return df


df_tickers = getStockExchangeCodes(apiKey)
l_tickers = df_tickers['code'].head(5).tolist()
print(l_tickers)

df_stockHistory = getStockMarketData(apiKey, l_tickers)

print(df_stockHistory)

#profile = df_stockHistory.profile_report(title='Stock Market Profiling Report')
#profile.to_file(output_file="output.html")



























#data_google = quandl.get("WIKI/GOOG")
#data_google = data_google.rename(columns={'Open': 'GOOGL_Open', 'High': 'GOOGL_High','Low':'GOOGL_Low','Close':'GOOGL_Close' ,'Volume':'GOOGL_Volume'})

#data_facebook = quandl.get("WIKI/FB")
#data_facebook = data_facebook.rename(columns={'Open': 'FB_Open', 'High': 'FB_High','Low':'FB_Low','Close':'FB_Close' ,'Volume':'FB_Volume'})


#data_msciworld = quandl.get("EUREX/FMWOM2017")
#data_msciworld = data_msciworld.rename(columns={'Open': 'MSCI_Open', 'High': 'MSCI_High','Low':'MSCI_Low','Close':'MSCI_Close' ,'Volume':'MSCI_Volume'})


# Merge the data on date
#data_all = pd.merge(data_allianz,data_google, how ='outer', left_index=True, right_index=True)
#data_all = pd.merge(data_all,data_facebook, how ='outer', left_index=True, right_index=True)
#data_all = pd.merge(data_all,data_msciworld, how ='outer', left_index=True, right_index=True)


#data_all['year'] = data_all.index.year
#data_all['month'] = data_all.index.month
#data_all['weekday'] = data_all.index.weekday







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