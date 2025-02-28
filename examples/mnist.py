import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from macrograd.model import Model, Linear, Optimizer
from macrograd.functions import relu, sigmoid, _to_var, log2
from macrograd import Tensor, e

np.random.seed(49)

def load_mnist_train(file_path) -> tuple[list, list]:
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise FileNotFoundError
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise Exception(e)

    labels = data.iloc[:, 0].values  # First column is labels
    images = data.iloc[:, 1:].values  # Remaining columns are pixel values

    return images, labels


def show_image(image: np.ndarray):
    return plt.imshow(image.reshape(28, 28))


def label2vec(label: int, num_classes: int = 10):
    vec = np.zeros((num_classes,))
    vec[label] = 1.0
    return vec


all_images, all_labels = load_mnist_train("../datasets/train.csv")

num_samples = len(all_images)
train_size = int(num_samples * 0.8)
test_size = num_samples - train_size
print(train_size)

train_images = np.array(all_images[:train_size])
train_labels = np.array(list(map(label2vec, all_labels[:train_size])))

test_images = np.array(all_images[train_size:])
test_labels = np.array(all_labels[train_size:])


def cross_entropy(y_true, y_pred):
    return -1 * (y_true * log2(y_pred)).sum() / y_true.arr.shape[0]


def softmax(x: Tensor):
    e_x = e**x
    return e_x / (e_x.sum(axis=1, keepdims=True))


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
        data = sigmoid(data)

        data = self.output_layer(data)
        data = softmax(data)

        return data


X = Tensor(train_images, requires_grad=False)
y = Tensor(train_labels, requires_grad=False)

model = MNIST_model(784, 10)
loss_ = []

parameters = model.init_params()

epochs = 200
learning_rate = 0.1
optimizer = Optimizer(learning_rate)

for _ in tqdm(range(epochs)):
    y_pred = model(X)

    loss = cross_entropy(y, y_pred)
    print(loss)

    parameters = optimizer.step(loss, parameters)

    loss_.append(loss.arr)

plt.plot(loss_)
plt.show()

# --- Accuracy Calculation ---
X_test = Tensor(test_images, requires_grad=False)
print(X_test.shape)
predictions = model(X_test)  # Get predictions on the test set
predicted_labels = np.argmax(
    predictions.arr, axis=1
)  # Convert probabilities to class labels


accuracy = np.mean(predicted_labels == test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
