import torch
import types
import torch.nn as nn
import torch.distributed as dist
from ..modifier import Modifier
from .utils import do_projection, apply_rope
import torch.nn.functional as F
from ..ops.flash_paged_topk import flash_paged_sparse_attn_func
from ..ops.all_gather import all_gather_uneven
from torch.utils.checkpoint import checkpoint


def get_tensor_parallel_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def get_tensor_parallel_rank():
    return dist.get_rank() if dist.is_initialized() else 0


class ColumnParallelLinearUneven(nn.Module):
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.world_size = get_tensor_parallel_world_size()
        self.rank = get_tensor_parallel_rank()

        if self.world_size <= 1:
            self.weight = nn.Parameter(linear_layer.weight.data.clone())
            self.bias = nn.Parameter(linear_layer.bias.data.clone()) if linear_layer.bias is not None else None
            return
        
        weight_chunks = torch.chunk(linear_layer.weight.data, self.world_size, dim=0)
        self.weight = nn.Parameter(weight_chunks[self.rank].clone())

        if linear_layer.bias is not None:
            bias_chunks = torch.chunk(linear_layer.bias.data, self.world_size, dim=0)
            self.bias = nn.Parameter(bias_chunks[self.rank].clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_parallel = F.linear(x, self.weight, self.bias)
        return output_parallel


class ColumnParallelLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.world_size = get_tensor_parallel_world_size()
        self.rank = get_tensor_parallel_rank()

        output_size_per_partition = linear_layer.out_features // self.world_size
        self.weight = nn.Parameter(
            linear_layer.weight.data[
                self.rank * output_size_per_partition: (self.rank + 1) * output_size_per_partition, :
            ].clone()
        )
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(
                linear_layer.bias.data[
                    self.rank * output_size_per_partition: (self.rank + 1) * output_size_per_partition
                ].clone()
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output_parallel = F.linear(x, self.weight, self.bias)
        return output_parallel


class RowParallelLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.world_size = get_tensor_parallel_world_size()
        self.rank = get_tensor_parallel_rank()

        input_size_per_partition = linear_layer.in_features // self.world_size
        self.weight = nn.Parameter(
            linear_layer.weight.data[
                :, self.rank * input_size_per_partition: (self.rank + 1) * input_size_per_partition
            ].clone()
        )
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output_parallel = F.linear(x, self.weight)
        
        if self.world_size > 1:
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
    
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
            
        return output_parallel
    

def model_forward(self, input_ids, kv_cache, grad_ckpt, **kwargs):
    hidden_states = self.model(input_ids, kv_cache, grad_ckpt)
    logits = self.lm_head(hidden_states)
    return logits


def model_model_forward(self, input_ids, kv_cache, grad_ckpt):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    for layer in self.layers:
        if grad_ckpt:
            hidden_states = checkpoint(layer, hidden_states, kv_cache, use_reentrant=False)
        else:
            hidden_states = layer(hidden_states, kv_cache)
        
    hidden_states = self.norm(hidden_states)
    return hidden_states


def layer_forward(self, hidden_states, kv_cache):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, kv_cache)
    hidden_states = residual.to(hidden_states.device) + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


def self_attn_forward(self, hidden_states, kv_cache):
    world_size = get_tensor_parallel_world_size()

    # =========================================
    stage = 2 if torch.is_grad_enabled() else 1
    kv_cache.visit(self.layer_idx, stage)
    # =========================================

    num_heads = self.config.num_attention_heads // world_size
    num_kv_heads = self.config.num_key_value_heads // world_size
    embed_dim = self.config.hidden_size
    head_dim = embed_dim // self.config.num_attention_heads

    # ===========================================
    past_length = kv_cache[self.layer_idx].num_kv
    if stage == 2:
        past_length -= hidden_states.shape[1]
    # ===========================================
        
    # position embedding
    pos = torch.arange(past_length, past_length + hidden_states.shape[1])
    pos = pos[None, :].to(hidden_states.device)
    cos, sin = self.rotary_emb(hidden_states, pos)

    # query & key & value projection
    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim, head_first=False)
    ques = apply_rope(ques, cos, sin)

    # ==========================================
    kv_cache[self.layer_idx].select(ques, stage)
    # ==========================================

    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim, head_first=False)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim, head_first=False)
    keys = apply_rope(keys, cos, sin)

    # ================================================
    kv_cache[self.layer_idx].update(keys, vals, stage)
    # ================================================

    attn_output = flash_paged_sparse_attn_func(
        ques,
        keys,
        vals,
        kv_cache[self.layer_idx])

    attn_output = attn_output.flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output


def mlp_forward(self, hidden_state):
    gate_output = self.gate_proj(hidden_state)
    up_output = self.up_proj(hidden_state)
    intermediate = self.act_fn(gate_output) * up_output
    
    return self.down_proj(intermediate)


class ModelForTraining(Modifier):
    def __init__(self, model, save_ckp: str, load_ckp: str, config: str):
        self.get_conf(config)
        
        self._parallelize_model(model)

        model.forward = types.MethodType(model_forward, model)
        model.model.forward = types.MethodType(model_model_forward, model.model)
        self.num_layers = len(model.model.layers)

        for layer in model.model.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)
            layer.mlp.forward = types.MethodType(mlp_forward, layer.mlp)

        if self.conf['lora']['enable']:
            model = self._init_lora(
                model, 
                lora_rank=self.conf['lora']['r'], 
                lora_alpha=self.conf['lora']['a'], 
                lora_dropout=self.conf['lora']['dropout'])

        super().__init__(model, save_ckp, load_ckp)

    def _parallelize_model(self, model):
        world_size = get_tensor_parallel_world_size()
        if world_size <= 1:
            return

        for layer in model.model.layers:
            # Attention
            layer.self_attn.q_proj = ColumnParallelLinear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = ColumnParallelLinear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = ColumnParallelLinear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = RowParallelLinear(layer.self_attn.o_proj)
            # MLP
            layer.mlp.gate_proj = ColumnParallelLinear(layer.mlp.gate_proj)
            layer.mlp.up_proj = ColumnParallelLinear(layer.mlp.up_proj)
            layer.mlp.down_proj = RowParallelLinear(layer.mlp.down_proj)

        # NOTE: modified
        # model.lm_head = ColumnParallelLinearUneven(model.lm_head)

    def _init_lora(self, model, lora_rank, lora_alpha, lora_dropout):
        target_modules = r".*\.(self_attn|mlp)\.(q_proj|v_proj|gate_proj|up_proj)"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules)
        return get_peft_model(model, peft_config)

    def _get_model(self):
        if self.conf['lora']['enable']:
            return self.model.model
        else:
            return self.model

    def ft_params(self):
        params = []
        for layer in self.model.model.layers:
            params.extend([
                layer.self_attn.q_proj.weight,
                layer.self_attn.k_proj.weight,
                layer.self_attn.v_proj.weight,
                layer.self_attn.o_proj.weight,
                layer.mlp.gate_proj.weight,
                layer.mlp.up_proj.weight,
                layer.mlp.down_proj.weight])
        params.append(self.model.lm_head.weight)
        return params

    def save_checkpoint(self, ckp_path: str):
        rank = get_tensor_parallel_rank()
        world_size = get_tensor_parallel_world_size()
        
        model = self._get_model()
        state_dict = model.state_dict()
        
        if rank == 0:
            print(f"Gathering weights and saving checkpoint to {ckp_path}...")
            full_state_dict = {}
        
        for key, param in state_dict.items():
            module_name, _, _ = key.rpartition('.')
            try:
                module = model.get_submodule(module_name)
            except AttributeError:
                module = None

            if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
                shards = [torch.empty_like(param) for _ in range(world_size)]
                dist.all_gather(shards, param.data)
                
                if rank == 0:
                    if isinstance(module, ColumnParallelLinear):
                        full_param = torch.cat(shards, dim=0)
                    else:
                        full_param = torch.cat(shards, dim=1)
                    full_state_dict[key] = full_param
            else:
                if rank == 0:
                    full_state_dict[key] = param.data

        if rank == 0:
            torch.save(full_state_dict, ckp_path)
            print(f"Checkpoint successfully saved.")

    def load_checkpoint(self, ckp_path: str):
        rank = get_tensor_parallel_rank()
        world_size = get_tensor_parallel_world_size()
        model = self._get_model()

        if rank == 0:
            print(f"Loading checkpoint from {ckp_path} on rank 0...")
            full_state_dict = torch.load(ckp_path, map_location='cpu')
        
        for key, param in model.named_parameters():
            module_name, _, _ = key.rpartition('.')
            try:
                module = model.get_submodule(module_name)
            except AttributeError:
                module = None

            if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
                if rank == 0:
                    full_param = full_state_dict[key]
                    if isinstance(module, ColumnParallelLinear):
                        shards = torch.chunk(full_param, world_size, dim=0)
                    else:
                        shards = torch.chunk(full_param, world_size, dim=1)
                    shards = [s.contiguous() for s in shards]
                else:
                    shards = None
                
                shard_to_load = torch.empty_like(param.data)
                dist.scatter(shard_to_load, scatter_list=shards, src=0)
                param.data.copy_(shard_to_load)
            else:
                if rank == 0:
                    full_param = full_state_dict[key]
                else:
                    full_param = torch.empty_like(param.data)
                
                dist.broadcast(full_param, src=0)
                param.data.copy_(full_param)
        
        if rank == 0:
            print("Model state loaded successfully across all ranks.")

    def forward(self, input_ids, labels, kv_cache, grad_ckpt=False):
        logits = self.model(
            input_ids=input_ids, 
            kv_cache=kv_cache, 
            grad_ckpt=grad_ckpt)
        
        # NOTE: modified
        # logits = all_gather_uneven(logits)

        if labels is not None:
            logits = logits.to(labels.device)
            logits = logits.squeeze(0)
            labels = labels.squeeze(0)
            return torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        else:
            return logits[:, -1:, :]

    @torch.no_grad()
    def generate(self, input_ids, tokenizer, max_new_tokens=128, eos_token_id=[2]):
        raise NotImplementedError
