import numpy as np
from .tensor import Tensor  # Make Tensor directly importable from macrograd
__all__ = ["Tensor"] # Optional: Control what gets imported with "from macrograd import *"

e = Tensor(np.e, requires_grad=False)
pi = Tensor(np.pi, requires_grad=False)
