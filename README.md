# Deep Learning Framework

As opossed to the great coders of [micrograd](https://github.com/karpathy/micrograd) or [tinygrad](https://github.com/tinygrad/tinygrad), since I am not as good as them, this framework shall be named **Macrograd**!

> Just tryping to create an autograd framework, like any great ML engineer should do I guess. My goal would be to keep implementing things as I am learning them in class.

## Features
- Eager Execution with Implicit Graph Creation
- pseudo-pytorch interface for model creation (with a more functional style)
- Numpy backend
- Stochastic Gradient Descent with/without Momentum
- MNIST with comparable pytorch performance

## Example

### Grap Execution
Everytime an operation is made on a `Tensor`, a new node is added into the implicit computational graph. To access the tensor data,
the whole graph needs to be computed, either by calling `realize()` on a `Tensor` or `Graph`.

A default graph is created is none is specified when calling `Tensor` to create new tensors or make operations on them:
```python
from macrograd import Tensor

def softmax(x: Tensor) -> Tensor:
    e_x = x.exp()
    return e_x / (e_x.sum(axis=1, keepdims=True))

my_tensor = Tensor(
    [[1, 2, 3, 4],
     [5, 6, 7, 8]])

logits = softmax(my_tensor)
logits.realize() # execture the graph

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
        data = relu(data)

        data = self.hidden_1(data)
        data = relu(data)

        data = self.hidden_2(data)
        data = relu(data)

        data = self.output_layer(data)
        data = softmax(data)

        return data

# init the model and create the parameters
model = MNIST_model(784, 10)
parameters = model.init_params()

epochs = 2
learning_rate = 0.01 # Stochastic Gradient Descent with Momentum
optimizer = SGD_MomentumOptimizer(learning_rate, alpha=0.99, params_copy=parameters)

# loop over all minibatches
for epoch in range(epochs):
    for X_batch, y_batch in tqdm(train_minibatches, desc=f"Epoch {epoch + 1}/{epochs}"):
        y_pred = model(X_batch) # forward pass

        # loss calculation
        loss = cross_entropy(y_batch, y_pred)

        # backward pass and parameter update
        parameters = optimizer.step(loss, parameters)
```
