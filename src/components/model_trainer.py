import os 
import sys
import dataclasses as dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier
)
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
# My custom models
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl') # The best model will be displayed on that file
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test,y_test=(
                train_array[:,:-1], # take every data without last column (X_train)
                train_array[:,-1], # take all the data from the last column (y_train)
                test_array[:,:-1], # take every data without last column (X_test)
                train_array[:,-1] # take all the data from the last column (y_test)
            )
            
        # Models i 'll use to classify the data
            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boost": GradientBoostingClassifier(),
                "xgboost":XGBClassifier(),
                "SVM": SVC(),
                "KNN": KNeighborsClassifier(), 
                'lightgbm':LGBMClassifier(),
                'LogisticRegression':LogisticRegression()
            }   
        # Create param grids for every and each of these algorithms
        # Decision Tree Hyperparameters
            params = {
            "Decision Tree" : {  
                'criterion':['gini', 'entropy'],
                'splitter':['best','random'],
                'max_depth':[2, 3, 5, 10],
                'min_samples_split':[5, 10, 20, 50, 100],
                'min_samples_leaf':[2,3,4,5,6,7,8,9]
            },

            # Random Forest Hyperparameters
            "Random Forest" : {
                'n_estimators':[100, 200],
                'max_depth':[2, 3, 5, 10],
                'min_samples_split':[5, 10, 20, 50, 100],
                'min_samples_leaf':[2,3,4,5,6,7,8,9],
                'bootstrap': [True, False]
                },

            # AdaBoost hyperparameters
            "AdaBoost" : {
                'n_estimators':[100, 200],
                'learning_rate':[0.01,0.1,0.2]
            },

            # Gradient Boosthyperparameters
            "Gradient Boost" :{
                'learning_rate':[0.01,0.1,0.2],
                'n_estimators':[100, 200],
                'min_samples_split':[5, 10, 20, 50, 100],
                'min_samples_leaf':[2,3,4,5,6,7,8,9]
            },

            # Xgboost hyperparameters
            "XGBoost" : {
                'n_estimators':[100, 200],
                'learning_rate':[0.01,0.1,0.2],
                'max_depth':[2, 3, 5, 10]
            },
            
            # SVM hyperparameters
            "SVM":{
                'C': [0.1, 1, 10, 100, 1000], # regularization parameter
                'kernel':['linear','poly','sigmoid'],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
            },

            # lightgbm hyperparameters
            'lightgbm': {
                'learning_rate':[0.01,0.1,0.2],
                'num_leaves':[10,11,12,13],
                'max_depth':[2, 3, 5, 10],
                'min_data_in_leaf':[2,3,4,5,6,7,8,9]
            },
          
            # Logistic Regression hyperparameters
            'LogisticRegression': {
                'penalty':['l2','l1','elasticnet'],
                'solver' : ['liblinear','saga'],
                'max_iter' : [100, 1000,2500, 5000],
                'C' : np.logspace(-4, 4, 20),
                'l1_ratio': [0, 0.5, 1]  # needed for elasticnet
            }
            }
            
            model_report: dict = evaluate_models(X_train=X_train,y_train=y_train,
                                                 X_test=X_test,y_test=y_test,models=models
                                                 ,param=params)
            ## To get best model score from dict
            best_model_score = max(model_report, key=lambda k: model_report[k][0])
            best_model = models[best_model_score][0]
            
            if best_model_score <0.6:
                raise CustomException("No model found")
            logging.info("Best model found for both training and testing data")
            
            save_object(file_path = self.model_trainer_config.trained_model_file_path,
                        obj= best_model)
            
            return model_report
            
        except Exception as e:
            raise CustomException(e,sys)