from dataclasses import dataclass
import os
import sys

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    
@dataclass
class DataTransformationConfig:
    preporcessor_path: str=os.path.join('artifacts',"preprocessor.pkl")
    pca_model_file_path = os.path.join("artifacts","pca.pkl")
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    pca_model_file_path = os.path.join("artifacts","pca.pkl")