'''
Author - Aditya Bhatt 8:05 AM 14-02-2023

Objective - 
1. dataframe Cleaning Module
'''

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        # Convert non-string values to empty strings
        dataframe['text'] = dataframe['text'].fillna('')

        # Convert text to lowercase
        dataframe['text'] = dataframe['text'].str.lower()

        # Remove HTML tags
        dataframe['text'] = dataframe['text'].apply(lambda x: re.sub(r'<.*?>', '', x))

        # Remove special characters and numbers
        dataframe['text'] = dataframe['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

        # Remove punctuation
        dataframe['text'] = dataframe['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

        # Remove URLs
        dataframe['text'] = dataframe['text'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        dataframe['text'] = dataframe['text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

        # Perform stemming
        stemmer = PorterStemmer()
        dataframe['text'] = dataframe['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x)]))

        # Handle white spaces
        dataframe['text'] = dataframe['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

        print("Data Preprocessing Done")
        return dataframe