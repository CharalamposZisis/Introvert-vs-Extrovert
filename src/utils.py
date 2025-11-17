# Functions to call when we want
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from src.exception import CustomException
import dill # module help us save pickle files
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    classification_report,
    confusion_matrix, 
    ConfusionMatrixDisplay
)
import shap

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path) # get the directory name of the specific path

        os.makedirs(dir_path,exist_ok=True) # make direction
        
        with open(file_path,'wb') as file_obj: # wb :write and binary
            dill.dump(obj,file_obj)  # obj:refers to the object we wanna pickle file_obj:the direction where we want to pickle out
    
    except Exception as e:
        raise CustomException(e,sys)
    
    
def evaluate_models(X_train,y_train,X_test,y_test, models,param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            # Make this grid search cv better 
                       
            # Parameter grid for GridSearchCV with cross validation
            gs = GridSearchCV(model,para,cv=5)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

           
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test) 
            
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            # Evaluate Performance methods
            
            acc_score = accuracy_score(y_test, y_test_pred)
            auc_score = roc_auc_score(y_test, y_pred_prob)
            
            # Classification report
            classification_re = classification_report(y_test, y_test_pred)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title(f'Confusion matrix for the Test data of {model}')
            plt.tight_layout
            plt.show() 
            
            # SHAP (SHapley Additive Explanation it help us to interpret model
            # predictions by making attributes of importance score)
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
            shap.initjs()
            shap_plot = shap.waterfall_plot(shap_values[0])
            
            
            report[list(models.keys())[i]] = acc_score, auc_score, classification_re , cm, shap_plot

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
            