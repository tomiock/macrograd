# Deep Learning Framework

As opossed to the great coders of [micrograd](https://github.com/karpathy/micrograd) or [tinygrad](https://github.com/tinygrad/tinygrad), since I am not as good as them, this framework shall be named **Macrograd**!

> Just tryping to create an autograd framework, like any great DL engineer should do I guess. My goal would be to keep implementing things that same useful to learn and to experiment with.

## Features
- Inferred Execution with explicit computational graphs
- Pseudo-pytorch interface for model creation (with a more functional style)
- Numpy backend
- Stochastic Gradient Descent with/without Momentum
- MNIST with comparable pytorch performance (model performance)

## Example

### Grap Execution
Everytime an operation is made on a `Tensor`, a new node is added into the implicit computational graph. To access the tensor data,
the whole graph needs to be computed, either by calling `realize()` on a `Tensor` or `Graph`.

A default graph is created if none is specified when calling `Tensor`. No need to handle manually handle a graph everytime:
```python
from macrograd import Tensor

# functions can be designed with the same notation that `numpy` uses:
def softmax(x: Tensor) -> Tensor:
    e_x = x.exp()
    return e_x / (e_x.sum(axis=1, keepdims=True))

def cross_entropy(y_true, y_pred) -> Tensor:
    return -1 * (y_true * (y_pred).log()).sum() / y_true.data.shape[0]


my_tensor = Tensor(
    [[1, 2, 3, 4],
     [5, 6, 7, 8]])

logits = softmax(my_tensor)
logits.realize() # execture the graph
# would be the same as `logits.graph.realize()

# access the computed tensor
logits.data
# >>> [[0.0320586  0.08714432 0.23688282 0.64391426]
# >>> [0.0320586  0.08714432 0.23688282 0.64391426]]
```

You access the default graph using `get_default_graph()`:
```python
from macrograd import Tensor
from macrograd.engine import get_default_graph


default_graph = get_default_graph()

my_tensor = Tensor(
    [[1, 2, 3, 4],
     [5, 6, 7, 8]])

logits = softmax(my_tensor)

# the whole graph is executed
default_graph.realize()

# access the computed tensor
logits.data
# >>> [[0.0320586  0.08714432 0.23688282 0.64391426]
# >>> [0.0320586  0.08714432 0.23688282 0.64391426]]
```

A custom graph can also be created by calling `Graph`, just passed as an argument into a tensor.
All successors of that tensor will be nodes on that graph.
```python
from macrograd import Tensor
from macrograd.engine import Graph

my_graph = Graph() # create a custom graph
my_tensor = Tensor(
    [[1, 2, 3, 4], [5, 6, 7, 8]],
    graph=my_graph, # pass the graph as an argument
)

logits = softmax(my_tensor)

# the whole graph is executed
my_graph.realize()

# access the computed tensor
logits.data
# >>> [[0.0320586  0.08714432 0.23688282 0.64391426]
# >>> [0.0320586  0.08714432 0.23688282 0.64391426]]
```

The graph can be visualized using `graphviz` by calling `Graph.visualized()` on any graph. This would be graph associated with the softmax function used before:

<img src="https://github.com/user-attachments/assets/b331564b-935e-4118-97f9-3d6fdcd13ff9" widht='600' height='600'>

All necessary information to execute the operations is saved on the graph, including the shape and the types of the tensors.


### MNIST
Multiclass MNIST Classification with 97% accuracy on test set with 2 or 1 epochs (60k train images)
```python
# data set is loaded from torchvision and minibatches are created
# see `example/mnist.py` for the rest of the code

# MODEL Creation like Pytorch
class MNIST_model(Model):
    def __init__(self, in_dims, out_dims):
        super(MNIST_model, self).__init__()

        self.input_layer = Linear(in_dims, 1024)
        self.hidden_1 = Linear(1024, 1024)
        self.hidden_2 = Linear(1024, 1024)
        self.output_layer = Linear(1024, out_dims)

    def __call__(self, data):
        data = self.input_layer(data)
        data = data.relu()

        data = self.hidden_1(data)
        data = data.relu()

        data = self.hidden_2(data)
        data = data.relu()

        data = self.output_layer(data)
        data = softmax(data)

        return data


model = MNIST_model(784, 10)

parameters = model.parameters()

epochs = 1
learning_rate = 0.001
optimizer = SGD_Momentum(parameters, learning_rate, alpha=0.8)

# this decorator needs to be used to set the limits of the computational graph
# macrograd will construct the graph on the first call, and freeze it.
# in the following calls no nodes will be added to the graph.
@compute_graph
def forward(x_batch, y_batch):
    y_pred = model(x_batch)
    loss = cross_entropy(y_batch, y_pred)

    return loss

# the actual forward pass needs to be defined by this type of decorated function.
# if not, the graph would keep growing with each forward pass call.

for _ in range(epochs):
    for x_batch, y_batch in train_minibatches:

        loss = forward(x_batch, y_batch)

        loss.backprop()

        optimizer.step(parameters)
        print(loss.data)
```

This would be the graph for this particular model, batch_size and data:
![computation_graph gv](https://github.com/user-attachments/assets/a58c6ae5-28e2-4df0-a4e2-e8b5e67bee76)


The ${{\color{Blue}\textsf{parameters in blue}}}\$, ${{\color{Red}\textsf{data in red}}}\$ and ${{\color{Green}\textsf{constansts in green}}}\$. The data nodes act like buffers, on each call they are filled with the input arrays that are provided to the `forward` function. During the first execution the graph is created according the initial data given, thus the batch size and tensor shape given as inputs needs must stay the same.

Note that the red nodes are the inputs given to the `forward` function, the `@compute_graph` decorator is in change of creating this nodes during the first call and of managing the buffers on subsequent calls.
