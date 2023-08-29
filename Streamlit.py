import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier 
from sklearn import naive_bayes
import streamlit as st
import tensorflow as tf
from MainF import data_1, training_data, voting, acc_hy,vector
from PIL import Image


st.title("Identification of fake reviews of product using Naive Bayes and decision tree based machine learning algorithms")


upload_file = st.file_uploader("Upload the Dataset", type=["csv"])

if upload_file==None:
    st.text("CHOOSE DATA")
else:
    st.write(data_1[['Review_text', 'Rev_Type']].head(10))
    st.text("----------------")
    st.text("HYBRID DECISION TREE AND NAIVE BAYES")
    st.text("----------------")
    st.text("PERFORMANCE ANALYSIS")
    st.write("Accuracy",acc_hy)
    

    
    st.text("Check Prediction")
    
    getin=st.text_input("Enter the Review")
    #st.write(getin)
    
    
    input_text = vector.transform([getin])

    pred=voting.predict(input_text)
    #print(pred)
    fina=st.button("PREDICT")
    #st.write(getin)
    #st.write(training_data1)
    tt=str(input_text)
    #st.write(tt)
    if fina:
        if tt=="":
           st.text("Please enter the valid review")

        elif pred==0:
            st.text("FAKE REVIEW")
       

        elif pred==1:
            st.text("REAL REVIEW")

    else:
        image = Image.open('OIP.jpg')
        st.image(image)  
        
