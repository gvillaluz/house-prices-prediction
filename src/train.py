from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib as jb
import pandas as pd

def train_model(model: XGBRegressor, training_data: pd.DataFrame): 
    print("Training Model.....")
    
    try:
       X = training_data.drop(columns='SalePrice', axis=1)
       y = training_data['SalePrice']
       
       X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.2, random_state=42)
       
       model.fit(X_train, Y_train)
       
       jb.dump(model, 'models/model.joblib')
       
    except Exception as e:
        print("Error in training: ", e)
        
def update_model():
    return