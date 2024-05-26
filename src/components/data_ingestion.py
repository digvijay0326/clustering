import os
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, 'C:/Users/digvi/OneDrive/Documents/ResoluteAi')
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import DataIngestionConfig

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Initiated")
        try:
            train_df = pd.read_excel(r"raw_data\train.xlsx")
            test_df = pd.read_excel(r"raw_data\test.xlsx")
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            
            train_df.to_csv(self.config.train_data_path, index=False, header=True)
            test_df.to_csv(self.config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion Completed Successfully")
            
            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()