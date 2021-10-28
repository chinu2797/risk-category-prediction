import pandas as pd
import re
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import SGDClassifier
import PySimpleGUI as sg
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer

from server.settings import BASE_DIR

stop_words = set(stopwords.words("english"))
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
cv=CountVectorizer(max_df=1.0,min_df=1, stop_words=stop_words, max_features=10000, ngram_range=(1,3))

def preprocess_text(sen):

    # Remove punctuations and numbers
    text = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)

    text = text.split()
    ##Stemming
    ps=PorterStemmer()    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  
            stop_words] 
    text = " ".join(text)

    return text

def predict(sentence,df,sgd):
    test=preprocess_text(sentence)
    test=test.lower()
    #print(test)
    test=[test] 
    tokenizer.texts_to_sequences(test)
    #new_words= tokenizer.word_index
    #print(word_index)
    cv.fit_transform(df['Observation'])
    test1=cv.transform(test)
    #print(test1)
    output=sgd.predict(test1)
    output=preprocess_text(output[0])
    return output

def retrain(master_df,X,y,sgd):
    X=preprocess_text(X)
    X=X.lower()
    data=[{'Observation': X,'Risk' : y}]
    master_df=master_df.append(data,ignore_index=True,sort=False)
    X=master_df['Observation']
    y=master_df['Risk']
    tokenizer.fit_on_texts(master_df['Observation'].values)
    #word_index = tokenizer.word_index
    X = cv.fit_transform(master_df['Observation'])
    #new_words=tokenizer.word_index
    sgd.fit(X,y)
    return sgd,master_df
    
    #print("Model trained on new data")