from .tcn import tcn

models = {
    "TCN": tcn()
}

def get_model(model_name, use_gpu=False, device=0, _type=1):
    
    return models[model_name]