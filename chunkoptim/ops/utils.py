import triton
from pygments.console import colorize


IS_BF16_ATOM_ADD_SUPPORTED = triton.__version__ >= "3.4.0"

if not IS_BF16_ATOM_ADD_SUPPORTED:
    print(colorize('yellow', "💩💩💩 BF16 atomic add is not supported by Triton < 3.4.0, please upgrade Triton to 3.4.0 or later. 💩💩💩"), flush=True)