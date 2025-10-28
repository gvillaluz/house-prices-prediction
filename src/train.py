from src.model import build_model
from src.data_preprocessing import get_data
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib as jb

def train_model(): 
    print("Training Model.....")
    
    try:
        X, y = get_data()
        model = build_model()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        
        jb.dump(model, 'models/model.joblib')
        
        print("Model Trained")
    except Exception as e:
        print("Error in training: ", e)

if __name__ == "__main__":
    train_model()