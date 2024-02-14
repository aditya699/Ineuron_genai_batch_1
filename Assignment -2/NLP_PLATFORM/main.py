import streamlit as st
from utils.data_loader import get_data
from preprocessing.data_preprocessing import clean_data
def main():
    st.title("Auto Sentiment Analysis")

    # User input for file path and title
    file_path = st.text_input("Enter the file path of the dataset:")
    file_path.replace('\\','/')
    # Load the dataset
    if st.button("Load Dataset"):
        try:
            data = get_data(file_path)
            st.write(data.head(1))
            st.write("Dataset loaded successfully!")
            data=clean_data(data)
            st.write("Dataset sucessfully Cleaned")
            st.write(data.head(1))
            
        except Exception as e:
            st.error(f"Contact the owner")
            
if __name__ == "__main__":
    main()
