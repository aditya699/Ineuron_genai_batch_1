'''
Author - Aditya Bhatt 8:05 AM 14-02-2023

Objective - 
1. dataframe Cleaning Module
'''
import streamlit as st
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
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from sklearn.svm import SVC

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

    # Vectorize text data
    vectorizer = CountVectorizer(binary=True)
    X_vec = vectorizer.fit_transform(X)  # Vectorize the entire dataset

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Build and train Multinomial Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = nb_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    print("Model training done")

    # Dump the trained model
    joblib.dump(nb_classifier, 'multinomial_nb_model.pkl')
    print("Model dumped")

    return accuracy




def preprocess_and_predict(test_data: pd.DataFrame, model_path: str, vectorizer: Union[CountVectorizer, None] = None) -> pd.DataFrame:
    """
    Preprocess the test data and make predictions using the provided model.
    """
    # Load the trained model
    model = joblib.load(model_path)
    # test_data.dropna(inplace=True)

    # Vectorize text data using binary count vectorizer
    if vectorizer is None:
        vectorizer = CountVectorizer(binary=True)
        vectorizer.fit(test_data['text'])  # Fit the vectorizer on the test data
    
    test_data_vec = vectorizer.transform(test_data['text'])

    # Make predictions
    predictions = model.predict(test_data_vec)

    # Add predictions to the DataFrame
    test_data['predictions'] = predictions

    test_data['predictions'].replace( {-1: 'negative', 0: 'neutral', 1: 'positive'},inplace=True)

    # Create a bar chart to visualize the distribution of sentiment predictions
    sentiment_counts = test_data['predictions'].value_counts()
    fig, ax = plt.subplots()
    colors = {'negative': 'red', 'neutral': 'yellow', 'positive': 'green'}
    sentiment_counts.plot(kind='bar', ax=ax, color=[colors.get(x, 'grey') for x in sentiment_counts.index])
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Reviews')
    ax.set_title('Distribution of Sentiment Predictions')
    st.pyplot(fig)

    return test_data

def word_2_vec_custom(data):
     # Tokenize the corpus
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in data['text']]

    # Train Word2Vec model
    model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

    # Save the trained model
    model.save("custom_word2vec.model")

    return None

def train_svm(data):

    # Load the trained Word2Vec model
    model = Word2Vec.load("custom_word2vec.model")
        # Extract text and labels
    X_text = data['text']
    y = data['sentiment']

    # Tokenize and compute average word vectors for each document
    X = []
    for text in X_text:
        tokens = word_tokenize(text.lower())
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            doc_vector = np.mean(vectors, axis=0)
            X.append(doc_vector)
        else:
            X.append(np.zeros(model.vector_size))  # Use zero vector for out-of-vocabulary words

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM classifier
    svm_classifier = SVC(kernel='linear')  # You can experiment with different kernels
    svm_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    # Dump the trained model
    joblib.dump(svm_classifier, 'svm_model.pkl')

    return accuracy

def preprocess_and_predict_svm(test_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the test data and make predictions using the provided model.
    """
    # Load the trained Word2Vec model
    model = Word2Vec.load("custom_word2vec.model")

    # Tokenize and compute average word vectors for each document
    X = []
    for text in test_data['text']:
        tokens = word_tokenize(text.lower())
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            doc_vector = np.mean(vectors, axis=0)
            X.append(doc_vector)
        else:
            X.append(np.zeros(model.vector_size))  # Use zero vector for out-of-vocabulary words

    # Load the trained SVM model from the fixed path
    svm_model = joblib.load("svm_model.pkl")

    # Make predictions
    predictions = svm_model.predict(X)

    # Add predictions to the DataFrame
    test_data['predictions'] = predictions

    # Map numeric predictions to sentiment labels
    sentiment_map = {-1: 'negative', 0: 'neutral', 1: 'positive'}
    test_data['predictions'] = test_data['predictions'].map(sentiment_map)

    # Create a bar chart to visualize the distribution of sentiment predictions
    sentiment_counts = test_data['predictions'].value_counts()
    fig, ax = plt.subplots()
    colors = {'negative': 'red', 'neutral': 'yellow', 'positive': 'green'}
    sentiment_counts.plot(kind='bar', ax=ax, color=[colors.get(x, 'grey') for x in sentiment_counts.index])
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Reviews')
    ax.set_title('Distribution of Sentiment Predictions')
    st.pyplot(fig)

    return test_data