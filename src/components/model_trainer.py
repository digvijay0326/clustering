import sys
import os
import numpy as np
from src.utils import save_object
from sklearn.cluster import KMeans
from src.config.configuration import ModelTrainerConfig
from src.logger import logging
from src.exception import CustomException

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
    
    def train_model(self, train_data, test_data):
        try:
            logging.info("Training started")
            kmeans = KMeans(n_clusters=4, random_state=42)
            kmeans.fit(train_data)
            logging.info("Training finished")
            clusters = kmeans.predict(train_data)
            save_object(self.config.trained_model_file_path, kmeans)
        
            np.save('artifacts/train_clusters.npy', kmeans.labels_)
            return clusters
        except Exception as e:
            raise CustomException(e, sys)
        