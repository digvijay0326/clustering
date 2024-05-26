import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, data):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            pca_path = os.path.join('artifacts', 'pca.pkl')
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            pca = load_object(pca_path)
            data_scaled = preprocessor.transform(data)
            data_pca = pca.transform(data_scaled)
            pred = model.predict(data_pca)
            
            return pred, data_pca
        except Exception as e:
            raise CustomException(e, sys)
    
class CustomData:
    def __init__(self, 
                    T1: int, T2: int, T3: int, T4: int, T5: int, T6: int, T7: int, T8: int, T9: int, T10: int,
                    T11: int, T12: int, T13: int, T14: int, T15: int, T16: int, T17: int, T18: int):
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4
        self.T5 = T5
        self.T6 = T6
        self.T7 = T7
        self.T8 = T8
        self.T9 = T9
        self.T10 = T10
        self.T11 = T11
        self.T12 = T12
        self.T13 = T13
        self.T14 = T14
        self.T15 = T15
        self.T16 = T16  
        self.T17 = T17
        self.T18 = T18
    def get_dataframe(self):
        try:
            data = {
                'T1': self.T1,
                'T2': self.T2,
                'T3': self.T3,
                'T4': self.T4,
                'T5': self.T5,
                'T6': self.T6,
                'T7': self.T7,
                'T8': self.T8,
                'T9': self.T9,
                'T10': self.T10,
                'T11': self.T11,
                'T12': self.T12,
                'T13': self.T13,
                'T14': self.T14,
                'T15': self.T15,
                'T16': self.T16,
                'T17': self.T17,
                'T18': self.T18}
            return pd.DataFrame(data, index=[0])
    
        except Exception as e:
            raise CustomException(e, sys)
    
