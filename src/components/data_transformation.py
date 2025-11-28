import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from src.exception import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from imblearn.pipeline import Pipeline as ImbPipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        """
        try:
            # Define columns
            numerical_columns = [
                'Time_spent_Alone', 
                'Social_event_attendance',
                'Going_outside',
                'Friends_circle_size',
                'Post_frequency'
            ]
            
            categorical_columns = ['Stage_fear','Drained_after_socializing']
            
            # Pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ("scaler", StandardScaler())
                ]
            )
            
            logging.info("Numerical columns standard scaling completed")
            
            # Pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Categorical columns encoding completed")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("categorical", cat_pipeline, categorical_columns),
                    ("numerical", num_pipeline, numerical_columns)
                ]
            )
            
            # Full pipeline with SMOTE for imbalanced data
            full_pipeline = ImbPipeline(
                steps=[
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=0))
                ]
            )

            return full_pipeline
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read Train and Test Data Completed")
            logging.info("Obtaining preprocessing Object")
            
            preprocessor_obj = self.get_data_transformer_object()
            
            target_column = 'Personality'
            
            # Drop target and 'id' from training features
            input_feature_train_df = train_df.drop(columns=[target_column, 'id'], axis=1)
            target_feature_train_df = train_df[target_column]
    
            # Test features: drop only target, keep 'id'
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info("Applying the preprocessing object on training and testing data")
            
            # Fit and resample training data
            X_train_resampled, y_train_resampled = preprocessor_obj.fit_resample(
                input_feature_train_df, target_feature_train_df
            )
            
            # Transform test data
            X_test_preprocessed = preprocessor_obj.named_steps["preprocessor"].transform(
                input_feature_test_df.drop(columns=['id'], axis=1)
            )
            
            # Concatenate features and target
            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr = np.c_[X_test_preprocessed, np.array(target_feature_test_df)]
            
            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            logging.info("Saved preprocessing object")
            
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
