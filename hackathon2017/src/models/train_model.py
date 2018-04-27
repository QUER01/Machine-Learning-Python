#################################################################################
#
#		ML algorithm: 			Random Forest
#		Author:					Julian Quernheim
#		Program Language:		    Python
#
#################################################################################


# Start Script
print("")
print("-------------------------------------------")
print("--     Starting train_model.py       ")
print("-------------------------------------------")
print("")



# Functions
def PCAData(w2vmodel):
    #get model, we use w2v only
    w2v=w2vmodel
 
    words_np = []
    #a list of labels (words)
    words_label = []
    for word in w2v.wv.vocab.keys():
        words_np.append(w2v[word])
        words_label.append(word)
    print('Added %s words. Shape %s'%(len(words_np),np.shape(words_np)))
 
    pca = PCA(n_components=50)
    pca.fit(words_np)
    reduced= pca.transform(words_np)
 
    return reduced





# Import libraries
import os 
from sklearn.externals import joblib
import pickle
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder

# Change working directory
path =os.getcwd()+'/'
os.chdir(path)
print(os.getcwd())
path  = "/home/osboxes/Documents/hackathon2017/"


# Load Variables
print('-')
print('- 1. Load Variables')
print('-')
f = open(path + 'model_variables.pckl', 'rb')
model_variables = pickle.load(f)
f.close()
model_type = model_variables[0][1]
model_name = model_variables[1][1]
feature_list = model_variables[2][1]
predictor = model_variables[3][1]


# Load data
print('-')
print('- 2. Load data')
print('-')
f = open(path + 'data/interim/X_train.pckl', 'rb')
X_train = pickle.load(f)
f.close()

f = open(path + 'data/interim/Y_train.pckl', 'rb')
X_train = pickle.load(f)
f.close()

f = open(path + 'data/interim/y.pckl', 'rb')
y = pickle.load(f)
f.close()

f = open(path + 'data/interim/X.pckl', 'rb')
X = pickle.load(f)
f.close()




# ML algorithm

#Assumed you have, X (predictor) and Y (target) for
#training data set and x_test(predictor) of test_dataset
#Create Random Forest object

if model_type == "Regression":
    print('-')
    print('- 3. Choosing RandomForestRegressor Model')
    print('-')
    from sklearn.ensemble import RandomForestRegressor
    model= RandomForestRegressor()
    
elif model_type == "Classifier":
    print('-')
    print('- 3. Choosing RandomForestClassifier Model')    
    print('-')
    from sklearn.ensemble import RandomForestClassifier
    model= RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    
elif model_type == "naive_bayes":
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()

#Train the model using the training sets and check score
print('-')
print('- 4. Fitting Model')
print('-')
classifier = model.fit(X,y)

#model_enc = OneHotEncoder()
#classifier = model.fit(model_enc.transform(model.apply(X)), y)




# Persist Model
print("Persisting model: Storing trained model to disk")
joblib.dump(model, path + 'models/model.pkl') 
joblib.dump(classifier, path + 'models/classifier.pkl') 
