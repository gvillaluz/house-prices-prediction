from xgboost import XGBRegressor
import pandas as pd

def predict_data(model: XGBRegressor, testing_data: pd.DataFrame):
    prediction = model.predict(testing_data)
    
    return prediction