from macrograd import Tensor

a = Tensor([1, 1, 1])
b = Tensor([2, 2, 2])

c = a + b
d = c / 10
e = d @ d
f = e.T
h = f.sqrt()
i = h.sum()
j = i.reshape(1, 1)

d.graph.visualize()
