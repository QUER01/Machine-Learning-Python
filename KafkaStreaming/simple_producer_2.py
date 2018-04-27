#data_producer.py
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_regression
import json
from kafka import KafkaProducer
from time import sleep
 

producer = KafkaProducer(
    bootstrap_servers='localhost:9091,localhost:9092')
 
#X, Y = make_blobs(n_samples=1200,n_features=2,centers=2, cluster_std=10.0)
#X, Y  = make_regression(n_samples=1000, n_features=2, n_informative=10, n_targets=1, bias=10.0, effective_rank=None, tail_strength=0.5, noise=10.0, shuffle=True, coef=False, random_state=None)


n_samples = 1000
n_outliers = 500


X, Y, coef = make_regression(n_samples=n_samples, n_features=2,
                                      n_informative=1, noise=100,bias =10, 
                                      coef=True, random_state=0)

# Add outlier data
np.random.seed(0)
X[:n_outliers] = 10 + 0.5 * np.random.normal(size=(n_outliers, 1))
Y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)




temp = np.zeros((X.shape[0],X.shape[1]+1))
temp[:,:X.shape[1]]=X
temp[:,X.shape[1]]=Y
for row in temp:
    message = json.dumps(dict([(i,x) for i,x in enumerate(row.tolist())]))
    print(message)
    message = str(message)
    binary_message = message.encode('utf-8')
    producer.send('simple-test2',binary_message)
    sleep(1)
