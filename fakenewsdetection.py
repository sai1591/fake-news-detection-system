#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:58:44 2022

@author: polinasaimanoj
"""
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

#importing dataset
df_fake = pd.read_csv("Fake.csv",)
df_true = pd.read_csv("True.csv")
print(df_fake.head(5))
print(df_true.head(5))

#Inserting a column "class" as target feature
df_fake["class"] = 0
df_true["class"] = 1
print(df_fake.shape)  
print(df_true.shape)

# Removing last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
    
    
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)
print(df_fake.shape)  
print(df_true.shape)

df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1 

print(df_fake_manual_testing.head(10))   
print(df_true_manual_testing.head(10))

df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")

#Merging True and Fake Dataframes
df_merge = pd.concat([df_fake, df_true], axis =0 )
print(df_merge.head(10))

df_merge.columns

#Removing columns which are not required

df = df_merge.drop(["title", "subject","date"], axis = 1)
print(df.isnull().sum())

#Random Shuffling the dataframe
df = df.sample(frac = 1)
print(df.head())

df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

print(df.columns)

print(df.head())

#Creating a function to process the texts
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text
df["text"] = df["text"].apply(wordopt)

#Defining dependent and independent variables
x = df["text"]
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

#Convert text to vectors
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#logestic Regression
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)
pred_lr=LR.predict(xv_test)
print(LR.score(xv_test, y_test))
print(classification_report(y_test, pred_lr))

#Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)

print(DT.score(xv_test, y_test))

print(classification_report(y_test, pred_dt))

#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)

pred_gbc = GBC.predict(xv_test)

print(GBC.score(xv_test, y_test))
print(classification_report(y_test, pred_gbc))

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)

pred_rfc = RFC.predict(xv_test)

print(RFC.score(xv_test, y_test))
print(classification_report(y_test, pred_rfc))

#Passive Aggresive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier

PAAG = PassiveAggressiveClassifier()
PAAG.fit(xv_train, y_train)

pred_paag = PAAG.predict(xv_test)
print(PAAG.score(xv_test, y_test))
print(classification_report(y_test, pred_paag))

#Naive bayes Multinomial
from sklearn.naive_bayes import MultinomialNB
CLF = MultinomialNB()
CLF.fit(xv_train, y_train)

pred_clf = CLF.predict(xv_test)
print(CLF.score(xv_test, y_test))
print(classification_report(y_test, pred_clf))


#Model Testing
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    pred_PAAG = PAAG.predict(new_xv_test)
    pred_CLF = CLF.predict(new_xv_test)
    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {} \nPAAG Predection: {} \nNBML Predection: {}".format(output_lable(pred_LR[0]),                                                                                                       output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0]),
                                                                                                              output_lable(pred_PAAG[0]),
                                                                                                              output_lable(pred_CLF[0])))
news = str(input())
manual_testing(news)










