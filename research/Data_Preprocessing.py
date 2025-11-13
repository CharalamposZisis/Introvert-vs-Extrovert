""" Data Preprocessing Stage
 Cleaning data with missing values
- Handling Outliers
- Scaling data
- Feature Encoding"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder

# import the training data
df = pd.read_csv('/home/xaris/Desktop/Projects/Introvert & Extrovert/Introvert-vs-Extrovert/Datasets/train.csv')
df.head()


class DataPreprocessingPipeline:
    def __init__(self, df, numerical, categorical, target='Personality'):
        self.df = df.copy()
        self.numerical = numerical
        self.categorical = categorical
        self.target = target

    def drop_id(self):
        if 'id' in self.df.columns:
            self.df.drop('id', axis=1, inplace=True)
        return self.df

    def impute_numerical(self):
        for col in self.numerical:
            self.df[col] = self.df[col].fillna(
                self.df.groupby(self.target)[col].transform('mean')
            )
        return self.df
    
    def impute_categorical(self):
        for col in self.categorical:
            self.df[col] = self.df[col].fillna(
                self.df.groupby(self.target)[col].transform('mode')
            )
        return self.df

    def handle_outliers(self):
        for column in self.numerical:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.df[column] = self.df[column].clip(lower=lower, upper=upper)
        return self.df

    def encode_target(self):
        self.df[self.target] = self.df[self.target].replace({'Extrovert': 1, 'Introvert': 0})
        return self.df

    
    def run_pipeline(self):
        """Runs all preprocessing steps in order."""
        self.drop_id()
        self.impute_numerical()
        self.handle_outliers()
        self.encode_target()
        return self.df