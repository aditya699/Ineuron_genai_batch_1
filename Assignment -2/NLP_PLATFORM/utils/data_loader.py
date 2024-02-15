'''
ETL Module
'''

import pandas as pd



def get_data(filepath: str) -> pd.DataFrame:
    '''
    General code to load the data
    '''
    try:
        data = pd.read_csv(filepath,encoding="ISO-8859-1")
        data=data[['text','sentiment']]
        print(f"Dataset loaded successfully from '{filepath}'\n")
        return data
    except Exception as e:
        print(f"Error loading dataset from '{filepath}': {str(e)}")
        return None
def get_data_test(filepath: str) -> pd.DataFrame:
    '''
    General code to load the data
    '''
    try:
        data = pd.read_csv(filepath,encoding="ISO-8859-1")
        data=data[['text']]
        print(f"Dataset loaded successfully from '{filepath}'\n")
        return data
    except Exception as e:
        print(f"Error loading dataset from '{filepath}': {str(e)}")
        return None

    