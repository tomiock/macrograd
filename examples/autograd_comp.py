from macrograd import Tensor
from macrograd.engine import get_default_graph


default_graph = get_default_graph()

def softmax(x: Tensor) -> Tensor:
    e_x = x.exp()
    return e_x / (e_x.sum(axis=1, keepdims=True))


my_tensor = Tensor(
    [[1, 2, 3, 4],
     [5, 6, 7, 8]])

logits = softmax(my_tensor)

default_graph.realize()
default_graph.visualize()

# access the computed tensor
logits.data
# >>> [[0.0320586  0.08714432 0.23688282 0.64391426]
# >>> [0.0320586  0.08714432 0.23688282 0.64391426]]
