from .tcn import tcn

models = {
    "TCN": tcn()
}

def get_model(model_name):
    return models[model_name]