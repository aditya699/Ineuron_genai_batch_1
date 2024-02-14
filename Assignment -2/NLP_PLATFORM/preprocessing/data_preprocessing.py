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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

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


def multinomial_nb(dataframe: pd.DataFrame) -> float:
    # Assign numerical values to sentiment categories
    sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
    dataframe['sentiment'] = dataframe['sentiment'].map(sentiment_map)

    # Splitting features and target labels
    X = dataframe['text']
    y = dataframe['sentiment']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text data
    vectorizer = CountVectorizer(binary=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Build and train Multinomial Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_vec, y_train)

    # Predictions
    y_pred = nb_classifier.predict(X_test_vec)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    print("Model training done")

    # Dump the trained model
    joblib.dump(nb_classifier, 'multinomial_nb_model.pkl')
    print("Model dumped")

    return accuracy