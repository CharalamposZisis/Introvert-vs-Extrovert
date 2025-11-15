# Functions to call when we want
import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
import dill # module help us save pickle files

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path) # get the directory name of the specific path

        os.makedirs(dir_path,exist_ok=True) # make direction
        
        with open(file_path,'wb') as file_obj: # wb :write and binary
            dill.dump(obj,file_obj)  # obj:refers to the object we wanna pickle file_obj:the direction where we want to pickle out
    
    except Exception as e:
        raise CustomException(e,sys)