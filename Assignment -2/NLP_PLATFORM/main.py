import streamlit as st
from utils.data_loader import get_data
from preprocessing.data_preprocessing import clean_dataframe

def main():
    st.title("Auto Sentiment Analysis")
    st.image('Officepic.JPG', width=200,caption="Author - Aditya Bhatt")
    st.markdown("---")
    st.write("Welcome to Auto Sentiment Analysis! This tool helps you automate Sentiment Analysis.")
    st.balloons()
    # User input for file path and title
    file_path = st.text_input("Enter the file path of the dataset:", "/path/to/your/dataset.csv")
    file_path.replace('\\', '/')

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
            
        except Exception as e:
            st.error("An error occurred while processing the dataset. Please check your file path or contact support.")

if __name__ == "__main__":
    main()
