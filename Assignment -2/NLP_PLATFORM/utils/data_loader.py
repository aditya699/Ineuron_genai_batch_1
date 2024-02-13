'''
ETL Module
'''
import logging
import pandas as pd

# Configure logging
logging.basicConfig(filename='etl.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data(filepath: str) -> pd.DataFrame:
    '''
    General code to load the data
    '''
    try:
        data = pd.read_csv(filepath)
        logging.info(f"Dataset loaded successfully from '{filepath}'")
    except Exception as e:
        logging.info(f"Error loading dataset from '{filepath}': {str(e)}")
    return data