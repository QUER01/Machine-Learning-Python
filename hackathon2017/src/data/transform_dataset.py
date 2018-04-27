

# Start Script
print("")
print("-------------------------------------------")
print("--     Starting transform_dataset.py       ")
print("-------------------------------------------")
print("")

# Import libraries
import os
import pandas as pd
from sklearn import preprocessing
import pickle
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

path =os.getcwd()+'/'
os.chdir(path)
print(os.getcwd())


path  = "/home/osboxes/Documents/hackathon2017/"

#Load data
df_train = pd.read_csv(path + 'data/interim/train.csv', error_bad_lines=False) # ,nrows=500000)
#df_test = pd.read_csv(path + "data/raw/test.csv", error_bad_lines=False)





#df_test = df_test.dropna()
#df_test = df_test.notnull()



# Load Variables
f = open(path + 'model_variables.pckl', 'rb')
model_variables = pickle.load(f)
f.close()

model_type = model_variables[0][1]
model_name = model_variables[1][1]
feature_list = model_variables[2][1]
predictor = model_variables[3][1]

df_train = df_train[feature_list + predictor]

df_train = df_train.dropna()
df_train = df_train[df_train.notnull()]


# Split data into features and predictor
X_train = df_train[feature_list]
Y_train = df_train[predictor]

#X_test = df_test[feature_list]





#Y_train = Y_train[1:1000]
#X_train =X_train[1:1000]

# Create the training data class labels
y = Y_train
    
# Create the document corpus list
corpus =  X_train
   

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

corpus_falttend = [val for sublist in corpus.values.tolist() for val in sublist]


# Clalculate distances
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus_falttend)
count_vect.vocabulary_.get(u'algorithm')

tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(X_train_counts)
X.shape


#tfidf = TfidfVectorizer(tokenizer=tokenize)
#X = tfidf.fit_transform(corpus_falttend)


# Create the training-test split of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


#Inpute missing data with mean
#imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
#X_train = pd.DataFrame(imp.fit_transform(X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")].index.values]))
#X_test = pd.DataFrame(imp.fit_transform(X_test[X_test.dtypes[(X_test.dtypes=="float64")|(X_test.dtypes=="int64")].index.values]))



# Create valdidation set (comment out if used in production)
#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)

# Store to pkl

f = open(path + 'data/interim/tfidf_transformer.pckl', 'wb')
pickle.dump(tfidf_transformer, f)
f.close()


f = open(path + 'data/interim/count_vect.pckl', 'wb')
pickle.dump(count_vect, f)
f.close()

f = open(path + 'data/interim/X_train.pckl', 'wb')
pickle.dump(X_train, f)
f.close()

f = open(path + 'data/interim/Y_train.pckl', 'wb')
pickle.dump(Y_train, f)
f.close()

f = open(path + 'data/interim/X_test.pckl', 'wb')
pickle.dump(X_test, f)
f.close()

f = open(path + 'data/interim/Y_test.pckl', 'wb')
pickle.dump(y_test, f)
f.close()

f = open(path + 'data/interim/y.pckl', 'wb')
pickle.dump(y, f)
f.close()

f = open(path + 'data/interim/X.pckl', 'wb')
pickle.dump(X, f)
f.close()

