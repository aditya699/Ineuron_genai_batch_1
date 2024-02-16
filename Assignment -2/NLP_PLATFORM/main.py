import streamlit as st
import pandas as pd
import joblib
from utils.data_loader import get_data,get_data_test
from preprocessing.data_preprocessing import clean_dataframe, multinomial_nb,preprocess_and_predict,word_2_vec_custom,train_svm,preprocess_and_predict_svm

def main():
    st.title("Auto Sentiment Analysis")
    st.image('Officepic.JPG', width=200, caption="Author - Aditya Bhatt")
    st.markdown("---")
    st.write("Welcome to Auto Sentiment Analysis! This tool helps you automate Sentiment Analysis.")
    st.balloons()

    st.write("Approach 1 is Naive Bayes with one hot encoding")
    
    # User input for file path of the dataset
    file_path = st.text_input("Enter the file path of the training dataset:", "/path/to/your/dataset.csv")
    file_path = file_path.replace('\\', '/')

    # User input for file path of the testing dataset
    test_file_path = st.text_input("Enter the file path of the testing dataset:", "/path/to/your/testing_dataset.csv")
    test_file_path = test_file_path.replace('\\', '/')

    # Load the dataset
    if st.button("Start Magic!"):
        try:
            data = get_data(file_path)
            st.success("Dataset loaded successfully!")
            st.subheader("Data Snapshot")
            st.write(data.head(5))

            st.info("Cleaning the dataset...")
            data = clean_dataframe(data)
            st.success("Dataset successfully cleaned!")
            st.subheader("Cleaned Data Snapshot")
            st.write(data.head(5))

            st.info("One hot Encoding with Naive Bayes Classifier is under process...")
            accuracy = multinomial_nb(data)
            st.success("Naive Bayes Classifier trained successfully!")
            st.subheader(f"The Model is trained and the accuracy is {accuracy}")

           
            # Read the testing data
            test_data = get_data_test(test_file_path)
            st.success("Testing DataFrame loaded successfully!")
            st.subheader("Testing Data Snapshot")
            st.write(test_data.head(5))

            # Preprocess the testing data
            test_data = clean_dataframe(test_data)
            st.success("Testing Dataset successfully cleaned!")
            st.subheader("Cleaned Data Snapshot")
            st.write(test_data.head(5))

            #Prediction on the same

            final_data=preprocess_and_predict(test_data,"multinomial_nb_model.pkl")
            st.subheader("Predictions on the same dataset successfully done!!")
            st.write(final_data)

            st.write("Approach 1 finised!!!!")

            st.write("Approach 2 is custom word2vec with SVM!!!")

            wor2vec=word_2_vec_custom(data)
            accuracy=train_svm(data)
            st.success(f"SVM with word2vec trained sucesfully with an accuracy of {accuracy}")
           
            final_data=preprocess_and_predict_svm(test_data)
            st.subheader("Predictions on the same dataset successfully done!!")
            st.write(final_data)

            st.write("Approach 2 is finished!!!")


        except Exception as e:
            st.error(f"An error occurred while processing the dataset. Please check your file path or contact support: {e}")

if __name__ == "__main__":
    main()
