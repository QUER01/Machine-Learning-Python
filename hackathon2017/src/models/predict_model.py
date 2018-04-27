
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

f = open(path + 'data/interim/Y_test.pckl', 'rb')
y_test = pickle.load(f)
f.close()

f = open(path + 'data/interim/tfidf_transformer.pckl', 'rb')
tfidf_transformer = pickle.load(f)
f.close()

f = open(path + 'data/interim/count_vect.pckl', 'rb')
count_vect = pickle.load(f)
f.close()


#Predict Output
y_pred = classifier.predict(X_test)
print(y_pred)


#docs_new = ['God is love well I dont know']
#X_new_counts = count_vect.transform(docs_new)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts)

y_pred = model.predict(X_test)








# Persist Model Prediction result
print("Persisting model prediction: Storing trained model prediction to disk")
joblib.dump(y_pred, 'models/y_pred.pkl')


if model_type in ("Classifier" , "naive_bayes"):
    
    # Model Evaluation for Classification problems
    print("Model Evaluation of Classification problems")
    
    # Clean up
    print("removing files of Model Evaluation of Classification problems first")
    
    #os.remove('models/y_pred.pkl')
    #os.remove('models/accuracy.pkl')
    #os.remove('models/ConfusionMatrix.pkl')
    #os.remove('models/logLoss.pkl')
    #os.remove('models/fpr.pkl')
    #os.remove('models/tpr.pkl')
    #os.remove('models/thresholds.pkl')
       
    # Model Evaluation
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred) 
    print("Accuracy:    "+ str(accuracy))
    # Persist Accuracy result
    print("Persisting model accuracy: Storing model accuracy to disk")
    joblib.dump(accuracy, 'models/accuracy.pkl')
     
    # Confusion Matrix
    ConfusionMatrix= confusion_matrix(y_test, y_pred)
    print("ConfusionMatrix:    "+ str(ConfusionMatrix))
    # Persist Confusion Matrix result
    print("Persisting model Confusion Matrix: Storing model Confusion Matrix to disk")
    joblib.dump(ConfusionMatrix, 'models/ConfusionMatrix.pkl')
    
    
    # log_loss
    logLoss= log_loss(y_test, y_pred)
    print("Log loss:    "+ str(logLoss))
    # Persist log_loss result
    print("Persisting model logLoss: Storing model logLoss to disk")
    joblib.dump(logLoss, 'models/logLoss.pkl')
    
    
    # ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=2)
    print("fpr:           "+ str(fpr))
    print("tpr:           "+ str(tpr))
    print("thresholds:    "+ str(thresholds))
    # Persist ROC result
    print("Persisting model Confusion Matrix: Storing model ROC to disk")
    joblib.dump(fpr, 'models/fpr.pkl')
    joblib.dump(tpr, 'models/tpr.pkl')
    joblib.dump(thresholds, 'models/thresholds.pkl')


elif model_type == "Regression":
    print("Model Evaluation for Regression problems")
    
    # Clean up
    print("removing files of Model Evaluation of Regression problems first")
    #os.remove('models/MAD.pkl')
    #os.remove('models/r_square.pkl')
    
     # Median absolute error
    from sklearn.metrics import median_absolute_error
    MAD = median_absolute_error(y_test, y_pred)
    print("MAD:           "+ str(MAD))
    # Persist MAD result
    print("Persisting model MAD: Storing model MAD to disk")
    joblib.dump(MAD, 'models/MAD.pkl')
    
    # R square
    from sklearn.metrics import r2_score
    r_square = r2_score(y_test, y_pred)
    print("r_square:           "+ str(r_square))
    # Persist MAD result
    print("Persisting model r_square: Storing model r_square to disk")
    joblib.dump(r_square, 'models/r_square.pkl')
    
    #explained_var
    from sklearn.metrics import explained_variance_score
    expalined_var = explained_variance_score(y_test, y_pred)
    print("expalined_var:           "+ str(expalined_var))
    # Persist MAD result
    print("Persisting model expalined_var: Storing model expalined_var to disk")
    joblib.dump(expalined_var, 'models/expalined_var.pkl')
    
    metrics = {
            "Regression_metrics": [
            {'Metric':'Mad','Value':str(MAD) },
            {'Metric':'r_square','Value':str(r_square) }
            
            ]
            }
    
    import json
    with open('app/static/data/metrics.json', 'w') as fp:
        json.dump(metrics, fp)
