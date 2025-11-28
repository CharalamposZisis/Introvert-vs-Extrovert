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
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param, shap_dir="shap_plots"):
    try:
        os.makedirs(shap_dir, exist_ok=True)
        report = {}

        for i, (model_name, model) in enumerate(models.items()):
            para = param[model_name]

            # GridSearchCV
            gs = GridSearchCV(model, para, cv=5)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Predict probability if available
            if hasattr(model, "predict_proba"):
                y_pred_prob = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_prob)
            else:
                y_pred_prob = None
                auc_score = None

            acc_score = accuracy_score(y_test, y_test_pred)
            classification_re = classification_report(y_test, y_test_pred)
            cm = confusion_matrix(y_test, y_test_pred)

            # Confusion Matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title(f'Confusion matrix for {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f"{model_name}_confusion_matrix.png"))
            plt.close()

            # SHAP explanations
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)

            # Αποθήκευση waterfall plot για πρώτο δείγμα
            shap_plot_path = os.path.join(shap_dir, f"{model_name}_shap_waterfall.png")
            if len(shap_values.shape) == 1:
                shap.plots.waterfall(shap_values[0], show=False)
            else:  # multi-output
                shap.plots.waterfall(shap_values[0, 0], show=False)
            plt.savefig(shap_plot_path)
            plt.close()

            report[model_name] = {
                "accuracy": acc_score,
                "auc": auc_score,
                "classification_report": classification_re,
                "confusion_matrix": cm,
                "shap_plot_path": shap_plot_path
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)

    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)