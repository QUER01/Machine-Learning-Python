
"""
Created on Sat Jun 24 16:41:54 2017

@author: root
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from functools import reduce
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
path  = "/home/osboxes/Documents/hackathon2017/"


def getSources():
    source_url = 'https://newsapi.org/v1/sources?language=en'
    response = requests.get(source_url).json()
    sources = []
    for source in response['sources']:
        sources.append(source['id'])
    return sources

def mapping():
    d = {}
    response = requests.get('https://newsapi.org/v1/sources?language=en')
    response = response.json()
    for s in response['sources']:
        d[s['id']] = s['category']
    return d

def category(source, m):
    try:
        return m[source]
    except:
        return 'NC'

def cleanData(path):
    data = pd.read_csv(path)
    data = data.drop_duplicates('url')
    data.to_csv(path, index=False)

def getDailyNews():
    sources = getSources()
    key = '66bef672eac2491f823da48d20cf9b73'
    url = 'https://newsapi.org/v1/articles?source={0}&sortBy={1}&apiKey={2}'
    responses = []
    prediction = []
    for i, source in tqdm(enumerate(sources)):
        try:
            u = url.format(source, 'top',key)
            response = requests.get(u)
            r = response.json()
            for article in r['articles']:
                print(article['title'])
                #---------------------------
                #     Prediction

                docs_new = [article['title']]
                X_new_counts = count_vect.transform(docs_new)
                X_new_tfidf = tfidf_transformer.transform(X_new_counts)

                y_pred = model.predict(X_new_tfidf)
                print("prediction" + str(y_pred))
                prediction.append(str(y_pred))
                
                #r = json.dump(str(y_pred) + str(docs_new))
                
                # Wait for 1 seconds
                #time.sleep(1)
                
               # ------------------
                   
                article['source'] = source
            responses.append(r)
            
        except:
            u = url.format(source, 'latest', key)
            response = requests.get(u)
            r = response.json()
            for article in r['articles']:
                print(article['title'])
                #---------------------------
                #     Prediction

                docs_new = [article['title']]
                X_new_counts = count_vect.transform(docs_new)
                X_new_tfidf = tfidf_transformer.transform(X_new_counts)

                y_pred = model.predict(X_new_tfidf)
                print("prediction" + str(y_pred))
                prediction.append(str(y_pred))
                
                #r = json.dump(str(y_pred) + str(docs_new))
                
                # Wait for 1 seconds
                #time.sleep(1)
                
               # ------------------
                   
                article['source'] = source
            responses.append(r)
      
    news = pd.DataFrame(reduce(lambda x,y: x+y ,map(lambda r: r['articles'], responses)))
    print(len(news))
    print(len(prediction))
    #prediction = [val for sublist in prediction for val in sublist] 
    news = pd.concat([news, pd.Series(prediction) ])
    
    news = news.dropna()
    news = news.drop_duplicates()
    d = mapping()
    news['category'] = news['source'].map(lambda s: category(s, d))
    news['scraping_date'] = datetime.now()

    try:
        aux = pd.read_csv(path + 'data/raw/news.csv')
    except:
        aux = pd.DataFrame(columns=list(news.columns))
        aux.to_csv(path + 'data/raw/news.csv', encoding='utf-8', index=False)

    with open(path + 'data/raw/news.csv', 'a') as f:
        news.to_csv(f, header=False, encoding='utf-8', index=False)

    cleanData(path + 'data/raw/news.csv')
    print('Done')

 
    
    
    

# Start Script
print("")
print("-------------------------------------------")
print("--     Starting predict_model.py       ")
print("-------------------------------------------")
print("")

# Import libraries
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
import time
import json
 
# Change working directory
path =os.getcwd()+'/'
os.chdir(path)
print(os.getcwd())


path  = "/home/osboxes/Documents/hackathon2017/"

# Load Variables
f = open(path  +'model_variables.pckl', 'rb')
model_variables = pickle.load(f)
f.close()

model_type = model_variables[0][1]
model_name = model_variables[1][1]
feature_list = model_variables[2][1]
predictor = model_variables[3][1]



# Load Model
print("Loading trained model from disk")
model = joblib.load(path + 'models/model.pkl') 
classifier = joblib.load(path + 'models/classifier.pkl') 


# Load data
print('-')
print('- 2. Load data')
print('-')

f = open(path + 'data/interim/X_test.pckl', 'rb')
X_test = pickle.load(f)
f.close()

f = open(path + 'data/interim/tfidf_transformer.pckl', 'rb')
tfidf_transformer = pickle.load(f)
f.close()

f = open(path + 'data/interim/count_vect.pckl', 'rb')
count_vect = pickle.load(f)
f.close()


if __name__ == '__main__':
    getDailyNews()
    
    


    
    
data = pd.read_csv(path + 'data/raw/news.csv')    
