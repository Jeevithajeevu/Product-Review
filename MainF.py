#========================= IMPORT PACKAGES ===========================

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn import naive_bayes
import streamlit as st
import tensorflow as tf

#==================== DATA SELECTION =========================


print("Analyse the Fake Product Review ")


print("-----------------------------------------")
print("============ Data Selection =============")
print("-----------------------------------------")
data=pd.read_csv("c:/Users/Harish/Desktop/ProductReview/Dataset.csv")
print(data.head(10))
print()
    
#================== PREPROCESSING =============================
    
#=== checking missing values ===
    
print("-----------------------------------------")
print("========= Check missing values  ======")
print("-----------------------------------------")
print(data.isnull().sum())
print()
    
data.drop_duplicates(inplace = True)
    
#=== drop unwanted columns ===
    
print("----------------------------------------------")
print("============= Drop the unwanted columns  =========")
print("----------------------------------------------")
print()
print("1.Before drop unwanted columns :",data.shape)
print()
print()
data_1=data.drop(['Unnamed: 0','Date'], axis = 1)
print("2.After drop unwanted columns  :",data_1.shape)
print()
print()

#========================= NLP TECHNIQUES ============================
    
#=== TEXT CLEANING ==== 
    
cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    return sentence
    
    
print("----------------------------------------------")
print("============ Before Applying NLP  ============")
print("----------------------------------------------")
print()
print(data['Review_text'].head(10))
    
print("----------------------------------------------")
print("============ After Applying NLP  =============")
print("----------------------------------------------")
print()
    
data_1['ReviewText'] = data_1['Review_text'].apply(cleanup) 

    
print(data_1['ReviewText'].head(10))




  

#========================= SENTIMENT ANALYSIS ==========================
    
#=== POS, NEG, NEUTRAL ===
    
analyzer = SentimentIntensityAnalyzer()
data_1['compound'] = [analyzer.polarity_scores(x)['compound'] for x in data_1['ReviewText']]
data_1['neg'] = [analyzer.polarity_scores(x)['neg'] for x in data_1['ReviewText']]
data_1['neu'] = [analyzer.polarity_scores(x)['neu'] for x in data_1['ReviewText']]
data_1['pos'] = [analyzer.polarity_scores(x)['pos'] for x in data_1['ReviewText']]
    
#======================= DATA SPLITTING ===========================
    
X = data_1["ReviewText"]
y = data_1['Rev_Type']
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
print("==============================================")
print("---------------- Data Splitting --------------")
print("==============================================")
print()
print("Total Number of review data : ", data_1.shape[0])
print()
print("Total Number of review data set for training : ", X_train.shape[0])
print()
print("Total Number of review data set for testing  : ", X_test.shape[0])
    
    
#==== tokenization ======
    
from tensorflow.keras.preprocessing.text import Tokenizer  #tokeniazation
    
#initialize the tokeniazation function
tokenizer = Tokenizer()
    
#fitting the model
tokenizer.fit_on_texts(data_1["ReviewText"])
X = tokenizer.texts_to_sequences(data_1["ReviewText"])
vocab_size = len(tokenizer.word_index)+1
    
print("===========================================")
print("Vocabulary size: {}".format(vocab_size))
print()
print("\nExample:\n")
print()
print("Sentence:\n{}".format(data_1["ReviewText"]))
print()
print("\nAfter tokenizing :\n{}".format(X[1]))
print()
    
    
#================= VECTORIZATION ================================
    
vector = CountVectorizer(stop_words = 'english', lowercase = True)
    
#fitting the data
training_data = vector.fit_transform(X_train)
    
#tranform the test data
testing_data = vector.transform(X_test)   
    
print("==============================================")
print("---------------- Vectorization --------------")
print("==============================================")
print()
print(training_data)
    
    
#================= CLASSIFICATION =================================
#Naive Bayes
    
print("----------------------------------------------")
print("=============== Naives Bayes =================")
print("----------------------------------------------")
print()
    
#initialize the model
Naive = naive_bayes.MultinomialNB()
    
#fitting the model
Naive.fit(training_data, y_train)
    
#predict the model
nb_pred = Naive.predict(testing_data)    
    
    
print()
print("Performances analysis for Naives bayes")
print()
acc_nb=accuracy_score(nb_pred,y_test)*100
print("1. Accuracy :",acc_nb,'%') 
print()
    
pre_nb=metrics.precision_score(y_test, nb_pred)*100
print("2. Precision :",pre_nb,'%')
print()
recall_nb=metrics.recall_score(y_test, nb_pred)*100
print("3. Recall :",recall_nb,'%')
print()
    
f1_nb=metrics.f1_score(y_test, nb_pred)*100
print("4. F1 score :",f1_nb,'%')
print()
    
# #decision tree
    
print("-----------------------------------------------")
print("=============== Decision Tree =================")
print("-----------------------------------------------")
print()
    
#initialize the model
dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=100, min_samples_leaf=1)
    
#fitting the model
dt.fit(training_data, y_train)
    
#predict the model
dt_prediction=dt.predict(testing_data)
    
print()
print("Performances analysis for decision tree")
print()
acc_dt=accuracy_score(y_test, dt_prediction)*100
print("1.Accuracy :",acc_dt,'%')
print()
pre_dt=metrics.precision_score(y_test, dt_prediction)*100
print("2. Precision :",pre_dt,'%')
print()
recall_dt=metrics.recall_score(y_test, dt_prediction)*100
print("3. Recall :",recall_dt,'%')
print()
f1_dt=metrics.f1_score(y_test, dt_prediction)*100
print("4. F1 score :",f1_dt,'%')    
    
    
    
# hybrid DT and NB 
    
from sklearn.ensemble import VotingClassifier
    
estimator = []
    
estimator.append(('NB', naive_bayes.MultinomialNB()))
estimator.append(('DT', DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=100, min_samples_leaf=1) ))
    
voting = VotingClassifier(estimators = estimator, voting ='soft')
voting.fit(training_data, y_train)
y_pred = voting.predict(testing_data)
    
acc_hy=accuracy_score(y_test, y_pred)*100
print("-----------------------------------------------")
print("HYBRID OF DECISION TREE AND NAIVE BAYES")
print("-----------------------------------------------")
print()
print("Accuracy :",acc_hy,'%')
print("-----------------------------------------------")    
print("Classification Report")
print(metrics.classification_report(y_test, y_pred))

#=== Visualization ====

data_1['length'] = data_1['Review_text'].apply(len)
plt.hist(data_1['length'],bins=30)
plt.title('Number of product reviews', fontsize=15)

data_1.groupby('Rev_Type').describe()
data_1.hist(column='length',by='Rev_Type',bins=30,color='green',figsize=(10,3))


colors = ['#FFD39B','#458B00']
plt.figure(figsize=(4,4))
label = data_1['Rev_Type'].value_counts()
plt.pie(label.values,colors = colors, labels=label.index, autopct= '%1.1f%%', startangle=90)
plt.title('Real and Fake Reviews Count', fontsize=15)


#CHECK TOP COMMON WORDS
words = '' 
for i in data_1["Review_text"]: 
    tokens = i.split()   
    words += " ".join(tokens)+" "

    
word_cloud = WordCloud(width = 700, height = 700, 
                       background_color ='white', 
                       min_font_size = 10).generate(words) 
plt.figure(figsize = (5, 5)) 
plt.imshow(word_cloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 


Perf = {'Decision_Tree':acc_dt, 'Naivebayes':acc_nb, 'Hybrid_model':acc_hy}
models = list(Perf.keys())
accvalues = list(Perf.values())
      
fig = plt.figure(figsize = (7, 5))
plt.bar(models, accvalues, color ='green', width = 0.4)
plt.ylabel("Accuracy")
plt.title("Performance Analysis")
plt.show(block=False)  

plt.show()

    
   