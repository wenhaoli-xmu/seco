def get_modifier(method: str, model_type):
    
    if method == 'blockwise':
        from .train_blockwise import ModelForTraining
        return ModelForTraining
    
    elif method == 'baseline':
        from .train_baseline import ModelForTraining
        return ModelForTraining
