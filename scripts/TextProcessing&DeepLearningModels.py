import pandas as pd
import numpy as np
import os
import json
import argparse
from pandas import json_normalize
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from langdetect import detect 
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from langdetect import detect
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop



def concat(x) :
    ecb =''
    fed = ''
    if x['ECB'] != []:
        ecb = x['ECB'][0]
    if x['FED'] != [] : 
        fed = x['FED'][0]
    return ecb + fed

def to_english(x):
  try :
    lang = detect(x)
    if lang == 'en':
      return str(x)
    else : 
      return ''
  except : 
      return ''

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) 
def clean_text(text):
    punct_tag=re.compile(r'[^\w\s]')
    text =punct_tag.sub(r'',text)
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text, re.UNICODE) 
    text = re.sub(r'\number','',text, re.UNICODE) 
    text=re.sub(r'[0-9]+', '',text, re.UNICODE)
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text,re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words] 
    text = " ".join(text)
    
    return text

def last_speech(x):
    for i in range(20):
        n = 19-i 
        if (x['J' + str(n)] != ''):
            return x['J'+str(n)]
        
def processing_speech(data) : 
    df = json_normalize(data)
    for i in range(20):
        df['J'+str(i)] = df.apply(lambda x : x['speech'][i], axis = 1)
    df['last speech'] = df.apply(lambda x : last_speech(x), axis = 1)
    df['last speech'] = df.apply(lambda x : concat(x['last speech']) , axis = 1)
    df['last speech'] = df.apply(lambda x : to_english(x['last speech']), axis = 1)
    df['last speech'] = df['last speech'].apply(lambda x: clean_text(x))
    
    df = df.drop('speech',axis = 1)
    
    return df 

def processing_stock(data) : 
    df = json_normalize(data)
    for i in range(20):
        df['J'+str(i)] = df.apply(lambda x : x['stock'][i], axis = 1)
    df = df.drop('stock',axis = 1)
    return df 
    
def tf(data) : 
    vect = TfidfVectorizer(max_features = 5000)
    tfidf_matrix = vect.fit_transform(data['last speech'])
    df = pd.DataFrame(tfidf_matrix.toarray(), columns = vect.get_feature_names())
    return(df)


def dayseven_try(data, train):
    
    stock_train = processing_stock(train)
    stock       = processing_stock(data)
    speech_train = processing_speech(train)
    speech       = processing_speech(data)
    
    
    X_classif       = speech_train.drop(['target_classif', 'target_reg'], axis = 1).reset_index(drop=True)
    X_total = pd.concat([X_classif, speech]).reset_index(drop=True)
    X_tfidf = tf(X_total)
    
    speech_train = X_tfidf.iloc[:len(train),:]
    speech      = X_tfidf.iloc[len(train):,:].reset_index(drop=True)
    
    stock_train = stock_train.drop('speech', axis = 1)
    stock       = stock.drop('speech', axis = 1)
    
    df_train    = pd.concat([stock_train, speech_train], axis =1)
    df          = pd.concat([stock, speech], axis = 1)
 
    X_train_classif      = df_train.drop(['target_classif','target_reg'] , axis = 1)
    X_test_classif       = df
    
    X_train_reg    = stock_train.drop(['target_classif','target_reg'], axis = 1)
    X_test_reg = stock 
    
    
    Y_train_classif = df_train['target_classif']
    Y_train_reg     = df_train['target_reg']
    
    
    #train classification
    randf       = RandomForestClassifier(n_estimators=200, max_depth=10,random_state=0)
    model_randf = randf.fit(X_train_classif, Y_train_classif)
    
    #train reg 
    scaler      = StandardScaler().fit(X_train_reg)
    X_train_reg = scaler.transform(X_train_reg)
    X_test_reg  = scaler.transform(X_test_reg)
    

    #Building the LSTM 
    STACKED_LSTM = Sequential()
    STACKED_LSTM.add(LSTM(units = 50,activation='relu',input_shape=(20,1),return_sequences=True))
    STACKED_LSTM.add(LSTM(units = 50,activation='relu'))
    STACKED_LSTM.add(Dense(1))


    #Compile the model
    opt = RMSprop(lr=0.0001)
    STACKED_LSTM.compile(optimizer = 'adam', loss = 'mse')

    #Fit STACKED_LSTM                 
    STACKED_LSTM.fit(X_train_reg,Y_train_reg,epochs = 100,verbose=0)
    pred_reg_inter = STACKED_LSTM.predict(X_test_reg)
    
    #pred
    pred_classif = model_randf.predict(X_test_classif)
    pred_reg = []
    for i in range(len(pred_reg_inter)):
        pred_reg.append(pred_reg_inter[i][0])
    return pred_reg, pred_classif
