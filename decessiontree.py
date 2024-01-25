#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:27:53 2022

@author: polinasaimanoj
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.metrics import confusion_matrix
from sklearn import tree
from wordcloud import WordCloud




#importing dataset
df_fake = pd.read_csv("Fake.csv",)
df_true = pd.read_csv("True.csv")


#Inserting a column "class" as target feature
df_fake["class"] = 0
df_true["class"] = 1


# Removing last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
    
    
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)

df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1 


df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")

#Merging True and Fake Dataframes
df_merge = pd.concat([df_fake, df_true], axis =0 )


df_merge.columns

#Removing columns which are not required

df = df_merge.drop(["title","date"], axis = 1)
print(df.isnull().sum())

#Random Shuffling the dataframe
df = df.sample(frac = 1)


df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)



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
#graph by subject
print(df.groupby(['subject'])['text'].count())
df.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()
#Graph by class
print(df.groupby(["class"])["text"].count())
df.groupby(["class"])["text"].count().plot(kind="bar")
plt.show()
#Word Cloud for Fake news

fake_data = df[df["class"] == 0]
all_words = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Word Cloud for True news
fake_data = df[df["class"] == 1]
all_words = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Most frequent words counter 
from nltk import tokenize
import nltk
token_space = tokenize.WhitespaceTokenizer();
def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()
# Most frequent words counter for fake news    
counter(df[df["class"] == 0], "text", 30)
# Most frequent words counter for True news  
counter(df[df["class"] == 1], "text", 30)

#Convert text to vectors
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#Decession Tree
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(random_state=1234)
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)
cm=confusion_matrix(y_test,pred_dt)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Fake', 'Predicted True'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
text_representation = tree.export_text(DT)
print(text_representation)




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
    pred_DT = DT.predict(new_xv_test)
    
    return print("\n\nDT Prediction: {} \nresult:{} \nTesting Accuracy : {:.2f} \n Confusion Matrix: \n\t\t\t{}" .format(output_lable(pred_DT[0]),
                                                                                                                classification_report(y_test, pred_dt),
                                                                                                                (accuracy_score(y_test,pred_dt)*100),
                                                                                                                cm))
                                                                                                                                                                                            
news = str(input())
manual_testing(news)