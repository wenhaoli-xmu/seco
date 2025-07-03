import torch
import time

class WallTime:
    def __init__(self, name: str, cuda: int = 0):
        self.name = name
        self.cuda_device = cuda
        self.times = []
        self.start_time = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.cuda_device)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.cuda_device)
        end_time = time.perf_counter()
        self.times.append(end_time - self.start_time)

    def result(self, detail: bool = False):
        if not self.times:
            print(f"Profiler '{self.name}': No times recorded.")
            return
            
        avg_time = (sum(self.times) / len(self.times))
        total_size_mb = float(self.name.split('(')[-1].split(' ')[0])
        bandwidth_gb_s = (total_size_mb / 1024) / avg_time

        print(f"Profiler '{self.name}':\t{avg_time:.4f} ms\t{bandwidth_gb_s:.4f} GB/s", flush=True)
        if detail:
            all_times_ms = [t * 1000 for t in self.times]
            print(f"  -> All runs (ms): {[f'{t:.4f}' for t in all_times_ms]}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        exit()

    num_kv_heads = 4
    head_dim = 128
    dtype = torch.bfloat16
    num_iterations = 5

    num_kv_cache_list = [10240 * (i + 1) for i in range(100)]

    print(f"Testing CPU to GPU transfer time using pinned memory and CUDA streams.")
    print("-" * 70)

    stream = torch.cuda.Stream()

    for num_kv_cache in num_kv_cache_list:
        tensor_shape = (1, num_kv_cache, num_kv_heads, head_dim)
        k_cache_cpu = torch.randn(tensor_shape, device='cpu', dtype=dtype).pin_memory()
        v_cache_cpu = torch.randn(tensor_shape, device='cpu', dtype=dtype).pin_memory()

        total_size_mb = (k_cache_cpu.nelement() * 2 * k_cache_cpu.element_size()) / (1024 * 1024)
        
        profiler_name = f'cpu-to-gpu: {num_kv_cache} tokens ({total_size_mb:.2f} MB)'
        profiler = WallTime(profiler_name, cuda=0)

        with torch.cuda.stream(stream):
            _ = k_cache_cpu.to('cuda', non_blocking=True)
            _ = v_cache_cpu.to('cuda', non_blocking=True)
        stream.synchronize()
        
        for _ in range(num_iterations):
            with profiler:
                with torch.cuda.stream(stream):
                    k_cache_gpu = k_cache_cpu.to('cuda', non_blocking=True)
                    v_cache_gpu = v_cache_cpu.to('cuda', non_blocking=True)
                
                stream.synchronize()

        profiler.result(detail=False)
