# -*- coding: utf-8 -*-
"""
Created on Wed oct 03 19:29:05 2017

@author: Julian Quernheim

purpose: A module for machine learning algorithms

TOC:
    0. hello
    1. summary_statistics 
    2. data_visualization
    
"""
   
# 1. Summary Statistics 
def summaryStatistics(df):
    # Dimensions of Dataset
    print(df.shape)
    
    # Sample
    print(df.head(20))
    
    # descriptions
    print(df.describe())
    
    # class distribution
    print(df.groupby('class').size())
    
    
def dataVisualization(df): 
    
    # Importing libraries
    print("Importing libaries:")
    import matplotlib
    import matplotlib.pyplot as plt
    print("Importing matplotlib version: " + matplotlib.__version__)
    
    import pandas
    from pandas.tools.plotting import scatter_matrix
    print("Importing pandas version: " + pandas.__version__)
    
    # box and whisker plots
    df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()
        
    # histograms
    df.hist()
    plt.show()
    
    # scatter plot matrix
    scatter_matrix(df)
    plt.show()
    
    
def trainModel(df):
    
    # Importing libraries
    print("Importing libaries:")
    import sklearn    
    from sklearn import model_selection
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    print("Importing sklearn version: " + sklearn.__version__)
    
    import matplotlib
    import matplotlib.pyplot as plt
    print("Importing matplotlib version: " + matplotlib.__version__)
    
    import pickle
    print("Importing pickle version: not accessible")
    
    
    
    # Split-out validation dataset
    array = df.values
    X = array[:,0:4]
    Y = array[:,4]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    
    
    # Test options and evaluation metric
    seed = 7
    scoring = 'accuracy'


    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    
    #pickle the model
    #import os
    #MY_DIR  = os.path.realpath(os.path.dirname(__file__))
    #PICKLE_DIR = os.path.join(MY_DIR, 'models')
    
    #fname = os.path.join(PICKLE_DIR, 'models')
    #with open(fname, 'rb') as f:
    #    pickle.dump(models, f)  
    
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    
    
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    