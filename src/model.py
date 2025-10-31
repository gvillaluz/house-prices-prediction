from xgboost import XGBRegressor

def build_model():
    return XGBRegressor(
        n_estimator=3000,
        learning_rate=0.02,
        max_depth=3
    )