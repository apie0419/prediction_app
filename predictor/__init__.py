from .tcn import tcn
from .xgboost import xgboost

models = {
    "TCN": tcn(),
    "XGBOOST": xgboost()
}

def get_model(model_name):
    
    return models[model_name]