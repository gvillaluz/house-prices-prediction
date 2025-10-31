from src.train import train_model, update_model
from src.data_preprocessing import get_data, get_new_data
from src.model import build_model
from src.predict import predict_data
import joblib as jb
import pandas as pd
import os

def main():
    try:
        model_path = 'models/model.joblib'
        new_data = 'data/new_data.csv'
        
        train_dataset, test_dataset = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')
        
        training_data, testing_data = get_data(train_dataset, test_dataset)
        
        if os.path.exists(model_path):
            model = jb.load(model_path)
            
            if os.path.exists(new_data):
                training_data = get_new_data(training_data)
                update_model(model, training_data)
                
        else:
            model = build_model()
            train_model(model, training_data)
            
        model = jb.load(model_path)
            
        prediction_data = predict_data(model, testing_data)
        
        print(prediction_data[0])
            
    except Exception as e:
        print("Error: ", e)

if __name__ == "__main__":
    main()