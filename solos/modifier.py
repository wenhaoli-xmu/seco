from abc import abstractmethod
from typing import Any
import torch
import json
from typing import List
from functools import partial
import logging


def segment(tensor, dim, n):
    total_length = tensor.shape[dim]

    for start in range(0, total_length, n):
        end = min(start + n, total_length)
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(start, end)
        yield tensor[tuple(indices)]


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


class Modifier(torch.nn.Module):
    def __init__(self, model, save_ckp, load_ckp):
        super().__init__()
        self.model = model
        self.load_ckp = load_ckp
        self.save_ckp = save_ckp

        if self.load_ckp is not None:
            self.load_checkpoint()

    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def get_conf(self, config):
        if config is not None:
            with open(config, 'r') as f:
                conf = json.load(f)
            print("=" * 40 + " Config " + "=" * 40)
            print(json.dumps(conf, indent=4
                ).replace("\n    ", "\n"
                ).replace("{", ""
                ).replace("}", ""
                ).strip().replace('"', ''))
            print("=" * 88)
            self.conf = conf
        else:
            self.conf = None
    

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad_(False)


    def unfreeze_model(self):
        for param in self.ft_params():
            param.requires_grad_(True)
    

    def load_checkpoint(self, ckp: str = None):
        ckp = ckp if ckp is not None else self.load_ckp
        checkpoint = torch.load(ckp, map_location="cpu")
        for param1, param2 in zip(self.ft_params(), checkpoint):
            param1.data = param2.data.to(device=param1.data.device, dtype=param1.data.dtype)


    def save_checkpoint(self, ckp: str = None):
        ckp = ckp if ckp is not None else self.save_ckp
        torch.save([maybe_zero_3(param) for param in self.ft_params()], ckp)


    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


    @abstractmethod
    def ft_params(self):
        raise NotImplementedError


    @abstractmethod
    def reset(self):
        raise NotImplementedError


class SegmentRecurrentModifier(Modifier):
    def __init__(self, model, save_ckp, load_ckp, chunk_size):
        super().__init__(model, save_ckp, load_ckp)
        self.chunk_size = chunk_size


    @abstractmethod
    def get_memories(self, segment_id):
        raise NotImplementedError
    
    
    @torch.no_grad()
    def generate_for_ae_task(
            self, 
            input_ids: torch.Tensor, 
            max_new_tokens: int = 256,
    ) -> List:
        assert input_ids.shape[1] == self.chunk_size

        self.model(input_ids=input_ids)
        logits = self.model(input_ids=torch.zeros((1,0), dtype=torch.long), prefix_repeat_token=True).logits
        inputs = torch.argmax(logits, dim=-1)
        while input_ids.shape[1] <= self.chunk_size + max_new_tokens:
            logits = self.model(input_ids=inputs).logits
            inputs = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, inputs.to(input_ids.device)], dim=-1)
        self.reset()
        return input_ids
    

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        eos_token_id: List = [2],
    ) -> List:

        prompt_length = input_ids.shape[1]
        context = input_ids[:,:-1]
        chunker = partial(segment, dim=-1, n=self.chunk_size)

        for chunk_context_inputs in chunker(context):
            self.model(input_ids=chunk_context_inputs)
        
        new_token = input_ids[:,-1:]
        while input_ids.shape[1] < prompt_length + max_new_tokens:
            logits = self.model(input_ids=new_token).logits.cpu()
            new_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, new_token.to(input_ids.device)], dim=1)

            # 在合适的时候进行打断
            if new_token.item() in eos_token_id:
                break

        self.reset()
        
        return input_ids
