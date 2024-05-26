from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path=train_path, test_path=test_path)
    model_trainer = ModelTrainer()
    clusters = model_trainer.train_model(train_data=train_arr, test_data=test_arr)
    print(clusters)