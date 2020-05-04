# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
sms_data=pd.read_csv("E:\\TEJAS\\EXCELR ASSIGMENTS\\COMPLETED\\NAVIES BAYIES\\SMS\\sms_raw_NB.csv",encoding="ISO-8859-1")
import re
######Here the given data is been cleaned in sense of removing commas,puncutations marks and lowering of alphabets.
def cleaning_text(i):
    i=re.sub("[^A-Za-z" "]+"," ",i).lower()
    i=re.sub("[0-9" "]+"," ",i)
    w=[]
    for word in i.split(" "):
        if len(word)>3:
           w.append(word)
    return (" ".join(w))
####Applying the cleaning funtion to whole dataset####
sms_data.text=sms_data.text.apply(cleaning_text)
sms_data=sms_data.loc [sms_data.text !=" ",:]
####Splitting the dataset into training and testing#########

from sklearn.model_selection import train_test_split
sms_train,sms_test=train_test_split(sms_data,test_size=0.3)

####Converting the datasets into matrix format###

def split_into_words(i):
    return[word for word in i.split(" ")]

sms_bow=CountVectorizer(analyzer=split_into_words).fit(sms_data.text)

all_sms=sms_bow.transform(sms_data.text)
all_sms.shape

train_sms_matrix=sms_bow.transform(sms_train.text)
train_sms_matrix.shape

test_sms_matrix=sms_bow.transform(sms_test.text)
test_sms_matrix.shape
#######Applying the navies bayes function########
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

classifier_mb=MB()
classifier_mb.fit(train_sms_matrix,sms_train.type)
train_sms_pred_mb=classifier_mb.predict(train_sms_matrix)
train_sms_accu_mb=np.mean(train_sms_pred_mb==sms_train.type)#98%

test_sms_pred_mb=classifier_mb.predict(test_sms_matrix)
test_sms_accu_mb=np.mean(test_sms_pred_mb==sms_test.type)#98%

classifier_gb=GB()
classifier_gb.fit(train_sms_matrix.toarray(),sms_train.type.values)
train_pred_gb=classifier_gb.predict(train_sms_matrix.toarray())
train_accu_gb=np.mean(train_pred_gb==sms_train.type.values)##90%

test_pred_gb=classifier_gb.predict(test_sms_matrix.toarray())
test_accu_gb=np.mean(test_pred_gb==sms_test.type.values)##81%


tfidf_transformer=TfidfTransformer().fit(all_sms)

train_sms_tfidf=tfidf_transformer.transform(train_sms_matrix)
train_sms_tfidf.shape

test_sms_tfidf=tfidf_transformer.transform(test_sms_matrix)
test_sms_tfidf.shape

classifier_mb=MB()
classifier_mb.fit(train_sms_tfidf,sms_train.type)
train_sms_pred_mb=classifier_mb.predict(train_sms_tfidf)
train_sms_tfidf_accu_mb=np.mean(train_sms_pred_mb==sms_train.type)##96%

test_sms_pred_mb=classifier_mb.predict(test_sms_tfidf)
test_sms_tfidf_accu_mb=np.mean(test_sms_pred_mb==sms_test.type)#95%

classifier_gb=GB()
classifier_gb.fit(train_sms_tfidf.toarray(),sms_train.type.values)
train_pred_gb=classifier_gb.predict(train_sms_tfidf.toarray())
train_accu_tfidf_gb=np.mean(train_pred_gb==sms_train.type.values)##90%

test_pred_gb=classifier_gb.predict(test_sms_tfidf.toarray())
test_accu_tfidf_gb=np.mean(test_pred_gb==sms_test.type.values)#81%

##Here we got 98% accuracy with mutlinomial Navies bayes###