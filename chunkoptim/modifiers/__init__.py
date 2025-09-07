def get_modifier(method: str, model_type):
    
    if method == 'blockwise':
        from .train_blockwise import ModelForTraining
    
    elif method == 'blockwise-tp':
        from .train_blockwise_tp import ModelForTraining

    elif method == 'blockwise-tp-sparse':
        from .train_blockwise_tp_sparse import ModelForTraining
    
    elif method == 'baseline':
        from .train_baseline import ModelForTraining

    elif method == 'baseline-tp':
        from .train_baseline_tp import ModelForTraining
    
    elif method == 'ringflash':
        from .train_ringflash import ModelForTraining

    elif method == 'yarn':
        from .train_yarn import ModelForTraining
    
    elif method == 'yarn-tp':
        from .train_yarn_tp import ModelForTraining

    elif method == 'yarn-tp-sparse':
        from .train_yarn_tp_sparse import ModelForTraining
        
    return ModelForTraining