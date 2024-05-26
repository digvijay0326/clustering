import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import DataTransformationConfig
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import zscore
from sklearn.decomposition import PCA
from src.utils import save_object

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
    
    def get_transformation_objet(self):
        try:
            processor = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )
            pca = Pipeline(
                steps=[
                    ('pca', PCA(n_components=2))
                ]
            )
            return processor, pca
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data Transformation Initiated")
        try:
            raw_train_df = pd.read_csv(train_path)
            raw_test_df = pd.read_csv(test_path)
            
            train_df = raw_train_df.drop(['target'], axis=1)
            preprocessing_obj, pca_obj = self.get_transformation_objet()
            zscores = zscore( train_df)
            outliers = np.where(abs(zscores) > 3.5)
            unique_outliers = np.unique(outliers)
            train_df = train_df.drop(unique_outliers, axis=0)
            train_df_scaled = preprocessing_obj.fit_transform(train_df)
            pca = pca_obj.fit_transform(train_df_scaled)
            scaled_clean_train =pca
            scaled_clean_test = preprocessing_obj.fit_transform(raw_test_df)
            scaled_clean_test = pca_obj.fit_transform(scaled_clean_test)
            
            save_object(self.config.preporcessor_path, preprocessing_obj)
            save_object(self.config.pca_model_file_path, pca_obj)
            np.save('artifacts/train_pca.npy', scaled_clean_train)
            return (
                scaled_clean_train,
                scaled_clean_test,
                self.config.preporcessor_path
            
            )
            
            logging.info("Data Transformation Completed Successfully")
        except Exception as e:
            raise CustomException(e, sys)