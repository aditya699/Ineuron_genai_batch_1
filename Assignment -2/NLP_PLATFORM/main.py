import streamlit as st
import logging
from utils import data_loader

logging.basicConfig(filename='main.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def main():
    st.title("Auto Sentiment Analysis")

    # User input for file path and title
    file_path = st.text_input("Enter the file path of the dataset:")
    file_path.replace('\\','/')
    # Load the dataset
    if st.button("Load Dataset"):
        try:
            data = data_loader.get_data(file_path)
            st.write("Dataset loaded successfully!")
            logging.info("Dataset loaded successfully!")

        except Exception as e:
            st.error(f"Contact the owner")
            logging.info(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    main()
