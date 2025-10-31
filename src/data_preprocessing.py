import pandas as pd
import numpy as np
import os

def process_data(data: pd.DataFrame):
    data_objects = data[data.select_dtypes(include='object').columns]
        
    data = data.drop(data_objects.isna().sum()[data_objects.isna().sum() > 500].index, axis=1)
    data_objects = data_objects.drop(data_objects.isna().sum()[data_objects.isna().sum() > 500].index, axis=1)
    
    data_objects = pd.get_dummies(data_objects)
    
    data_numerical = data[data.select_dtypes(exclude='object').columns]
    
    numerical_null = data_numerical.columns[data_numerical.isna().any()]
    
    for col in numerical_null:
        data_numerical.loc[:, col] = data_numerical[col].fillna(np.round(data_numerical[col].mean()))
        
    data = pd.concat([data_numerical, data_objects], axis=1)
    
    return data

def get_data(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:   
        test['SalePrice'] = np.nan
        
        data = pd.concat([train, test], axis=0)
        
        data = data.set_index("Id")
        
        cleaned_data = process_data(data)
        
        training_data = cleaned_data[: len(train)]
        testing_data = cleaned_data[len(train) :]
        
        testing_data = testing_data.drop(columns='SalePrice', axis=1)
        
        return training_data, testing_data
    
    except Exception as e:
        print("Error in data preprocessing: ", e)
        
def get_new_data(data: pd.DataFrame):
    if os.path.exists('../data/new_data.csv'):
        new_data = pd.read_csv('../data/new_data.csv')
        
        new_training_data = pd.concat([data, new_data], axis=0)
        
        return new_training_data