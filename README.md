# Deep Learning Framework

As opossed to the great coders of [micrograd](https://github.com/karpathy/micrograd) or [tinygrad](https://github.com/tinygrad/tinygrad), since I am not as good as them, this framework shall be named **Macrograd**!

> Just tryping to create an autograd framework, like any great ML engineer should do I guess. My goal would be to keep implementing things as I am learning them in class.

## Features
- pseudo-pytorch interface for model creation (with a more functional style)
- Numpy backend
- Stochastic Gradient Descent with/without Momentum
- MNIST with comparable pytorch performance

## Example

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
