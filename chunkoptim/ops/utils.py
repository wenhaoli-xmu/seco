import triton
from pygments.console import colorize

IS_BF16_ATOM_ADD_SUPPORTED = triton.__version__ >= "3.4.0"
