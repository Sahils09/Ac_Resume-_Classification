# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:47:43 2023

@author: My Acer
"""

# Import necessary libraries

import pickle
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
nltk.download('wordnet')


# Load the trained machine learning model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
    
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


html_temp = """ 
    <div style ="background-color:pink;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Resume Classification ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True)

# Define function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if not word in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# Define the Streamlit app
def app():
    st.title('Resume Classifier')
    user_input = st.text_area('Enter your resume text here:', height=350)
    if user_input:
        preprocessed_input = preprocess_text(user_input)
        X_test = vectorizer.transform([preprocessed_input])
        category = model.predict(X_test)
# Add a button to trigger the classification
        if st.button('Classify'):
            if category == 0:
                st.write('This resume is classified as:', 'Peoplesoft Resume')
            elif category == 1:
                st.write('This resume is classified as:', 'ReactJS Developer Resume')
            elif category == 2:
                st.write('This resume is classified as:', 'SQL Developer Resume')
            else:
                st.write('This resume is classified as:', 'Workday Resume')
            
    

if __name__ == '__main__':
    app()



