from macrograd import Tensor
from macrograd.tensor import Node, Graph

a = Tensor(1)
b = Tensor(1)

c = a + b

d = c * c

print(c)
print(d)
