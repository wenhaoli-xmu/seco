from ..modifier import Modifier
import torch


class Greedy(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        super().__init__(model, save_ckp, load_ckp)


    def ft_params(self):
        return []

    
    def reset(self):
        pass


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, eos_token_id=[2], prof=None):

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)[None, :]

        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)

        # put the tensor on to the model's device
        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        # prefilling
        output = self.model(input_ids=input_ids)
        logits, past_key_values = output.logits, output.past_key_values
        new_tok = logits.argmax(dim=-1)
        new_ids = [new_tok]

        # generation
        new_tok = input_ids[:, -1:]
        new_ids = []

        with prof if prof is not None else do_nothing():
            while len(new_ids) < max_new_tokens:
                output = self.model(input_ids=new_tok, past_key_values=past_key_values)
                logits, past_key_values = output.logits, output.past_key_values

                new_tok = logits.argmax(dim=-1)
                if new_tok.ravel().item() in eos_token_id: break
                new_ids.append(new_tok.ravel().item())

        new_ids = torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device)[None, :]
        return torch.cat([input_ids, new_ids], dim=-1)

