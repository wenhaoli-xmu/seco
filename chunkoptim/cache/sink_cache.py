import torch


class SinkCacheManager:
    def __init__(
            self, 
            batch_size, 
            num_kv_heads, 
            head_dim, 
            sink: int = 16,
            sliding_window: int = 256,
            prealloc: int = None):
        
        if prealloc is not None:
            assert torch.is_grad_enabled() is False, f"Only support KV cache pre-allocation in inference mode."

        super().__init__()
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.prealloc = prealloc
        self.sink = sink
        self.sliding_window = sliding_window
        self.reset()

    def reset(self):
        self.num_kv = 0

        if self.prealloc is not None:
            self.key = torch.empty((self.batch_size, self.prealloc, self.num_kv_heads, self.head_dim), dtype=torch.bfloat16, device='cuda')
            self.val = torch.empty((self.batch_size, self.prealloc, self.num_kv_heads, self.head_dim), dtype=torch.bfloat16, device='cuda')

        else:
            self.key = None
            self.val = None

    def update(self, key, val):
        update_token = key.shape[1]

        if self.prealloc is not None:
            beg_pos = self.num_kv
            end_pos = self.num_kv + update_token
            assert end_pos <= self.max_tokens, f"Context length exceed its maximum supported value."
            self.key[:, beg_pos: end_pos] = key
            self.val[:, beg_pos, end_pos] = val
            self.num_kv = end_pos
        
        else:
            self.key = key
            self.val = val
            self.num_kv = update_token
