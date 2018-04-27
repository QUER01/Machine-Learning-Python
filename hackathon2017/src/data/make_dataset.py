# -*- coding: utf-8 -*-

# Start Script
print("")
print("-------------------------------------------")
print("--     Starting make_dataset.py            ")
print("-------------------------------------------")
print("")

# Import libraries
import os
import pandas as pd

# Change working directory
#path ='/media/ventum/Work USB 1/99_Others/Fortbildung/Kaggle/2017_Sberbank Russian Housing Market/SHB/'
path =os.getcwd()+'/'

path  = "/home/osboxes/Documents/hackathon2017/"

os.chdir(path)
print(os.getcwd())

# Load Trainngs and Test Data
df_train = pd.read_csv(path + "data/raw/train.csv" , error_bad_lines=False , nrows =20000)
df_train['predictor'] = pd.Series(1, index=df_train.index)



# Get google news
from bs4 import BeautifulSoup
import requests

n= 0
url="https://news.google.co.in/"
#url = "https://newsapi.org/v1/articles?source=google-news&sortBy=top&apiKey=66bef672eac2491f823da48d20cf9b73"
code=requests.get(url)
soup=BeautifulSoup(code.text,'html5lib')

df_google = pd.DataFrame(columns=['uuid', 'title'])
for title in soup.find_all('span',class_="titletext"):
    n = n+1
    df_google = df_google.append(pd.Series([n, title.text], index=['uuid','title']), ignore_index=True)
    
df_google['predictor'] = pd.Series(0, index=df_google.index)


# Load Trainngs and Test Data
df_news_real = pd.read_csv(path + "data/raw/real_news.csv" , error_bad_lines=False , nrows =20000)
df_news_real['predictor'] = pd.Series(0, index=df_news_real.index)
df_news_real.columns.values[1] = 'title'   


#df_test = pd.read_csv("data/raw/test.csv", error_bad_lines=False)
#df_macro = pd.read_csv("data/raw/macro.csv", error_bad_lines=False)

#from sklearn.datasets import fetch_20newsgroups
#newsgroups_train = fetch_20newsgroups(subset='train')



# Remove NAN



df_train = df_train.append(df_google)
df_train = df_train.append(df_news_real)



# Store data frame in folder
df_train.to_csv(path + 'data/interim/train.csv' ,encoding ='utf8')

