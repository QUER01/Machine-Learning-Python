#################################################################################
#
#		ML algorithm: 			Random Forest
#		Author:					Julian Quernheim
#		Program Language:		Python
#
#################################################################################


# Start Script
print("")
print("-------------------------------------------")
print("--     Starting train_model.py       ")
print("-------------------------------------------")
print("")

# Import libraries
import os 
import numpy as np
from sklearn.externals import joblib
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from nltk.stem.porter import PorterStemmer
import pandas as pd
import gensim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt




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


def plotWords(modelw2v):
    #get model, we use w2v only
    w2v=modelw2v
 
    words_np = []
    #a list of labels (words)
    words_label = []
    for word in w2v.wv.vocab.keys():
        words_np.append(w2v[word])
        words_label.append(word)
    print('Added %s words. Shape %s'%(len(words_np),np.shape(words_np)))
 
    pca = PCA(n_components=2)
    pca.fit(words_np)
    reduced= pca.transform(words_np)
 
    # plt.plot(pca.explained_variance_ratio_)
    for index,vec in enumerate(reduced):
        # print ('%s %s'%(words_label[index],vec))
        if index <100:
            x,y=vec[0],vec[1]
            plt.scatter(x,y)
            plt.annotate(words_label[index],xy=(x,y))
    plt.show()
    return reduced






# Load Variables
f = open('model_variables.pckl', 'rb')
model_variables = pickle.load(f)
f.close()
model_type = model_variables[0][1]
model_name = model_variables[1][1]
feature_list = model_variables[2][1]
predictor = model_variables[3][1]




if model_type == "Regression":
    from sklearn.ensemble import RandomForestRegressor
elif model_type == "Classifier":
    from sklearn.ensemble import RandomForestClassifier



# Change working directory
path =os.getcwd()+'/'
#path ='/run/media/manjaro/Work USB 1/99_Others/Fortbildung/Kaggle/2017_Sberbank Russian Housing Market/SBB/'
os.chdir(path)
print(os.getcwd())
path  = "/home/osboxes/Documents/DST/python/"


#########################################
# Model Fake
#########################################

# Load data
X_train = pd.read_csv(path +'data/interim/X_train.csv', delimiter=",")
Y_train = pd.read_csv(path +'data/interim/Y_train.csv',delimiter=',')

Y_train = Y_train[1:100]
X_train =X_train[1:100]


token_dict = []
lines = []
from nltk.tokenize import word_tokenize

for line in X_train.ix[:,1].values:
    line = str(line)
    #token_dict.append(word_tokenize(line))
    lines.append(line)
#print(token_dict)


#Cleaning
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in lines]        







#Document Term Matrix
# Importing Gensim
import gensim
from gensim import corpora
from gensim.models import AuthorTopicModel




# Create predictor vector
author2doc = dict()
author2doc = X_train.ix[:,2]


# Create a dictionary representation of the documents, and filter out frequent and rare words.

from gensim.corpora import Dictionary
dictionary = Dictionary(doc_clean)

# Remove rare and common tokens.
# Filter out words that occur too frequently or too rarely.
max_freq = 0.5
min_wordcount = 20
dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)
_ = dictionary[0]

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Vectorize data.
# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in doc_clean]



model = AuthorTopicModel(corpus=corpus, num_topics=10, id2word=dictionary.id2token, author2doc=author2doc, chunksize=2000, passes=1, eval_every=0, iterations=1, random_state=1)






#Model
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=2, id2word = dictionary, passes=50)





#Run Model
print(ldamodel.print_topics(num_topics=2, num_words=3))









# let X be a list of tokenized texts (i.e. list of lists of tokens)
model_fake = gensim.models.Word2Vec(token_dict, size=100, window=5, min_count=5, workers=4, hs=1, negative=0)
#model = gensim.models.Doc2Vec(lines,dm = 0, alpha=0.1, size= 20, min_alpha=0.025)

# Test the model
model_fake.score(["The fox jumped over a lazy dog".split()])
model_fake.wv.most_similar(positive=['fake', 'news'], negative=['man'])
model_fake.n_similarity(['fake', 'real'], ['eat', 'restaurant'])



pca_data = plotWords(model_fake)

pca_data = PCAData(model_fake)
pca_data

predictor_fake = np.ones((len(pca_data),1)) 
data_fake = np.hstack((pca_data,predictor_fake))




#########################################
# Model Real
#########################################

# Load data
X_train = pd.read_csv(path +'data/interim/X_train_real.csv', delimiter=",")
Y_train = pd.read_csv(path +'data/interim/Y_train_real.csv',delimiter=',')

Y_train = Y_train[1:100]
X_train =X_train[1:100]


token_dict = []
lines = []
from nltk.tokenize import word_tokenize

for line in X_train.values:
    line = str(line)
    token_dict.append(word_tokenize(line))
    lines.append(line)
print(token_dict)


# let X be a list of tokenized texts (i.e. list of lists of tokens)
model_real = gensim.models.Word2Vec(token_dict, size=100, window=5, min_count=5, workers=4, hs=1, negative=0)
#model = gensim.models.Doc2Vec(lines,dm = 0, alpha=0.1, size= 20, min_alpha=0.025)

# Test the model
model_real.score(["The fox jumped over a lazy dog".split()])
model_real.wv.most_similar(positive=['fake', 'news'], negative=['man'])
model_real.n_similarity(['fake', 'real'], ['eat', 'restaurant'])



pca_data = PCAData(model_real)
pca_data

predictor_real = np.zeros((len(pca_data),1)) 
data_real = np.hstack((pca_data,predictor))

# ML algorithm

#Assumed you have, X (predictor) and Y (target) for
#training data set and x_test(predictor) of test_dataset
#Create Random Forest object

if model_type == "Regression":
    model= RandomForestRegressor()
elif model_type == "Classifier":
    model= RandomForestClassifier()
#Train the model using the training sets and check score

model.fit(pca_data, predictor)



# Persist Model
print("Persisting model: Storing trained model to disk")
joblib.dump(model, path + 'models/trained_model.pkl') 





