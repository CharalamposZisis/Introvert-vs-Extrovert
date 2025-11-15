import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig: # for this we could create a separate folder where i could determine where to save all of these
    train_data_path: str = os.path.join('artifacts','train.csv') # input data and i tell the system where to store
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # take the values of the above class 
    
    def initiate_data_ingestion(self): # on this function is the first stage where we start pull the data practically
        logging.info("Entered the data ingestion method or component")
        try:
            df1 = pd.read_csv('Datasets/train.csv') # here we could start read our data either from mongodb etc (now just from csv file)
            df2 = pd.read_csv('Datasets/test.csv')
            logging.info('Read the datasets as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # make directory where the train data will saved 
            df1.to_csv(self.ingestion_config.train_data_path)
            df2.to_csv(self.ingestion_config.test_data_path)
            
            logging.info('Train test & test initiated')
            train_set = df1
            test_set = df2
            
            train_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Ingestion of the data is completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data =obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)