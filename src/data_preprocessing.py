import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        print("Processing the data...")
        data = pd.read_csv("./data/car_prices.csv")
        
        X = data.drop(columns='Sell Price($)')
        y = data['Sell Price($)']
        
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        
        encoded = encoder.fit_transform(data[['Car Model']])
        
        car_models = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Car Model']))
        
        X = pd.concat([X.drop(columns='Car Model'), car_models], axis=1)
        
        return X, y

    except Exception as e:
        print("Error in data preprocessing: ", e)