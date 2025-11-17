import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from imblearn.pipeline import Pipeline as ImbPipeline

@dataclass
class DataTransformationConfig: # it will give me any path/inputs for my data transformation i am gonna use
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl') # that 's the file that we will create
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() # take the values of the above class 
    
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        """
        try:
            # Define columns
            numerical_columns = ['id', # drop this column
                         'Time_spent_Alone', 
                         'Social_event_attendance',
                         'Going_outside',
                         'Friends_circle_size',
                         'Post_frequency']
            
            categorical_columns = ['Stage_fear','Drained_after_socializing']
            
            # Don 't forget to drop the id column
            class columnDropperTransformer():
                def __init__(self,columns):
                    self.columns=columns

                def transform(self,X,y=None):
                    return X.drop(self.columns,axis=1)

                def fit(self, X, y=None):
                    return self 

            # Create a pipeline for numerical values    
            num_pipeline = Pipeline(
                steps=[
                    ('columnDropper',columnDropperTransformer(['id'])),
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ("scaler", StandardScaler())
                ]
            )
            
            logging.info("Numerical columns standard scaling completed")
            # Create a pipeline for categorical feature
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            ) 
            
            logging.info("Categorical columns encoding completed")
            
                
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            preprocessor = ColumnTransformer(transformers=
                [
                    ("categorical",cat_pipeline,categorical_columns),
                    ("numerical", num_pipeline, numerical_columns)
                ]
            )
            
            # We use ImbPipeline since this 
            full_pipeline = ImbPipeline( 
                steps = [
                    ('preprocessor',preprocessor),
                    ('smote', SMOTE(random_state=0))  # with smote we handle the imbalanced data
                ]
            )

            # We want to return the preprocess from the last above lines
            return full_pipeline
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read Train and Test Data Completed")
            logging.info("Obtaining preprocessing Object")
            
            # Object to get access on the above function
            preprocessor_obj = self.get_data_transformer_object()
            
            # Define the target column
            target_column = 'Personality'
            
            # Define the exact training data
            input_feature_train_df = train_df.drop(columns = [target_column],axis =1)
            target_feature_train_df = train_df[target_column]
            
            # Define the exact test data
            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]
            
            logging.info("Apllying the preprocessing object on training and testing dataframe")
            
            X_train_resampled, y_train_resampled = preprocessor_obj.fit_resample(input_feature_train_df, target_feature_train_df)
            X_test_preprocessed = preprocessor_obj.named_steps["preprocessor"].transform(input_feature_test_df)

            
            # This function c_ is used to concatenate input feature (train array) and target (input train_df)
            train_arr = np.c_[
                X_train_resampled,np.array(y_train_resampled)
            ]
            
            # The input feature test arr to is already array
            test_arr = np.c_[
                X_test_preprocessed,np.array(target_feature_test_df)
            ]
            
            # Save the preprocessor object (pickle) using the custom save_object from utils
            save_object( 
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj = preprocessor_obj
            )        
                                 
            logging.info("Saved preprocessing object")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)