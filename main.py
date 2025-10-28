from src.train import train_model
import joblib as jb

def main():
    try:
        model = jb.load('models/model.joblib')
        
            
    except Exception as e:
        print("Error: ", e)

if __name__ == "__main__":
    train_model()
    main()