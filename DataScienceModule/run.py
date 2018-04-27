# -*- coding: utf-8 -*-
"""
Created on Wed oct 03 19:29:05 2017

@author: Julian Quernheim
    
"""
import pandas
from DataScienceModule import ml_module


# Load dataset


# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pandas.read_csv(url, names=names)


ml_module.summaryStatistics(df)
ml_module.dataVisualization(df)
ml_module.trainModel(df)