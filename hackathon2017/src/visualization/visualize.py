# Start Script
print("")
print("-------------------------------------------")
print("--     Starting visualize.py       ")
print("-------------------------------------------")
print("")

# Import libraries
import os
from sklearn.externals import joblib
from sklearn.metrics import  auc
import pickle



# Change working directory
path =os.getcwd()+'/'
os.chdir(path)
print(os.getcwd())




# Load Variables
f = open('model_variables.pckl', 'rb')
model_variables = pickle.load(f)
f.close()
model_type = model_variables[0][1]
model_name = model_variables[1][1]
feature_list = model_variables[2][1]
predictor = model_variables[3][1]




if model_type == "Classifier":
    import matplotlib.pyplot as plt
    print("Visualize model Evaluation of Classification problems")
    # Load Model
    print("Loading trained model prediction from disk")
    model = joblib.load('models/y_pred.pkl')
    accuracy = joblib.load('models/accuracy.pkl')
    ConfusionMatrix = joblib.load('models/ConfusionMatrix.pkl')
    model = joblib.load('models/logLoss.pkl')
    fpr = joblib.load('models/fpr.pkl')
    tpr = joblib.load('models/tpr.pkl')
    thresholds = joblib.load('models/thresholds.pkl')
    
    
    # Calculate the area under the curve
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)'% roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('reports/figures/ROC.png')

elif model_type == "Regression":
    print("Visualize model Evaluation of Regression problems")
    
    # Load model metrics
    MAD = joblib.load('models/MAD.pkl')
    print("MAD:           "+ str(MAD))
    
    r_square  = joblib.load('models/r_square.pkl')
    print("r_square:           "+ str(r_square))
    
    expalined_var  = joblib.load('models/expalined_var.pkl')
    print("expalined_var:           "+ str(expalined_var))
    
    
