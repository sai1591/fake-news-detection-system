#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 19:22:37 2022

@author: polinasaimanoj
"""

from PyQt5 import QtCore, QtGui, QtWidgets
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
from sklearn import metrics



class Ui_Form2(object):
    def __init__(self,text1):
              self.text1=text1
    def copy(self):
          
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
           df = df_merge.drop(["title", "subject","date"], axis = 1)
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
           print(plt.show())
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
   
    
    
    
                    return  self.label_2.setText("\n\nDecision Tree Prediction: {} \nresult:\n{} \n\nTesting Accuracy : {:.2f} \n\nConfusion Matrix: \n{}" .format(output_lable(pred_DT[0]),
                                                                                                                classification_report(y_test, pred_dt),
                                                                                                                (accuracy_score(y_test,pred_dt)*100),
                                                                                                                cm))
                                                                                                      
                                                                                                              
           news = str(self.text1)
           manual_testing(news)

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(621, 512)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(140, 10, 301, 31))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setObjectName("label")
        self.textBrowser = QtWidgets.QTextBrowser(Form)
        self.textBrowser.setGeometry(QtCore.QRect(60, 60, 491, 371))
        self.textBrowser.setObjectName("textBrowser")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(70, 70, 471, 341))
        self.label_2.setText("")
        self.label_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(210, 440, 161, 51))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.copy()
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Decession Tree"))
        self.label.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:24pt; font-weight:600; color:#2194ef;\">DECESSION TREE</span></p></body></html>"))
        self.pushButton.setText(_translate("Form", "Graphs"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form2()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())