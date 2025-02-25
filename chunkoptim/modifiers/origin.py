from ..modifier import Modifier


class Origin(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        super().__init__(model, save_ckp, load_ckp)


    def ft_params(self):
        return []

    
    def reset(self):
        pass


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
