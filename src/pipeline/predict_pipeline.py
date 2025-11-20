import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
        
    ### CHECK THE TYPES AND IF THEY ARE CORRECT GENERALLY
class CustomaData:
    def __init__( self,
        Time_spent_Alone: int, 
        Social_event_attendance: int,
        Going_outside: int,
        Friends_circle_size:int ,
        Post_frequency: int ,
        Stage_fear : bool,
        Drained_after_socializing: bool,
        Personality:bool
    ):
    
    self.Time_spent_Alone = Time_spent_Alone
    
    self.Social_event_attendance = Social_event_attendance
    
    self.Going_outside = Going_outside
    
    self.Friends_circle_size = Friends_circle_size
    
    self.Post_frequency = Post_frequency
    
    self.Stage_fear = Stage_fear
    
    self.Drained_after_socializing = Drained_after_socializing
    
    self.Personality = Personality