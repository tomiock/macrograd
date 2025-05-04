from .tensor import Tensor  # Make Tensor directly importable from macrograd
from .engine import Graph, get_default_graph

# TODO: handle constants

#e = Tensor(np.e, requires_grad=False)
#pi = Tensor(np.pi, requires_grad=False)
