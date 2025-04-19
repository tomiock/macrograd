import numpy as np
from macrograd import Tensor

a = Tensor([[1, 2], [3, 4]])

inner = a.T

inner.realize()

print(inner.data)
inner.graph.visualize()
