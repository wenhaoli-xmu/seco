def get_modifier(method: str, model_type):

    if method == 'origin':
        from .origin import Origin
        return Origin
    
    elif method == 'train':
        from .train import ModelForTraining
        return ModelForTraining
    
    elif method == 'train-ckpt':
        from .train_ckpt import ModelForTraining
        return ModelForTraining
