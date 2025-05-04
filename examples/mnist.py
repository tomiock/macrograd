import numpy as np
from tqdm import tqdm

from macrograd.model import Model, Linear
from macrograd.optimizers import SGD_Momentum
from macrograd.functions import relu
from macrograd import Tensor

import torchvision
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MNIST_dataset(Dataset):
    def __init__(self, data, partition="train"):
        self.data = data
        self.partition = partition
        print("total len", len(self.data))
        self.images = np.empty((len(self.data), 28 * 28), dtype=np.float32)
        self.labels = np.empty((len(self.data), 10), dtype=np.float32)
        self._preprocess_data()

    def __len__(self):
        return len(self.data)

    def from_pil_to_tensor(self, image):
        return torchvision.transforms.ToTensor()(image)

    def __getitem__(self, idx):
        # Return preprocessed data.
        return {"img": self.images[idx], "label": self.labels[idx]}

    def _preprocess_data(self):
        """Preprocesses the entire dataset and stores it in memory."""
        for idx in tqdm(
            range(len(self.data)), desc=f"Preprocessing {self.partition} data"
        ):
            image = self.data[idx][0]
            image_tensor = self.from_pil_to_tensor(image)
            image_tensor = image_tensor.view(-1).numpy()

            label = torch.tensor(self.data[idx][1])
            label = F.one_hot(label, num_classes=10).float().numpy()

            self.images[idx] = image_tensor
            self.labels[idx] = label


def create_minibatches(dataset, batch_size, num_samples=None):
    if not num_samples:
        num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)  # Shuffle the data
    minibatches = []

    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_images = dataset.images[batch_indices]
        batch_labels = dataset.labels[batch_indices]

        X_batch = Tensor(batch_images, requires_grad=False, precision=np.float32)
        y_batch = Tensor(batch_labels, requires_grad=False, precision=np.float32)
        minibatches.append((X_batch, y_batch))

    return minibatches


def evaluate_model(model, test_minibatches):
    correct_count = 0
    total_count = 0

    for X_batch, y_batch in test_minibatches:
        predictions = model(X_batch)
        predicted_labels = np.argmax(predictions.data, axis=1)
        true_labels = np.argmax(y_batch.data, axis=1)
        correct_count += np.sum(predicted_labels == true_labels)
        total_count += 10

    accuracy = correct_count / total_count
    return accuracy


def softmax(x: Tensor):
    e_x = x.exp()
    return e_x / (e_x.sum(axis=1, keepdims=True))


def cross_entropy(y_true, y_pred):
    return -1 * (y_true * log2(y_pred)).sum() / y_true.data.shape[0]


train_set = torchvision.datasets.MNIST(".data/", train=True, download=True)
test_set = torchvision.datasets.MNIST(".data/", train=False, download=True)

train_dataset = MNIST_dataset(train_set, partition="train")
test_dataset = MNIST_dataset(test_set, partition="test")

batch_size = 10
num_samples = None
train_minibatches = create_minibatches(
    train_dataset, batch_size, num_samples=num_samples
)
test_minibatches = create_minibatches(test_dataset, batch_size)


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


mlflow.set_tracking_uri("http://atenea:5000")

model = MNIST_model(784, 10)

parameters = model.init_params()

epochs = 10
learning_rate = 0.02
optimizer = SGD_Momentum(parameters, learning_rate, alpha=0.8)
# scheduler = LinearScheduler(total_iter=8, target_lr=.0001)

hyper_params = {
    "epochs": epochs,
    "learning_rate": learning_rate,
    "momentum": 0.8,
    "batch_size": batch_size,
    "number_samples": num_samples,
    "optimizer": optimizer.__class__.__name__,
    "scheduler": None,
}

loss_history = []
batch_loss_history = []
lr_history = []

mlflow.set_experiment("mnist-macrograd")


def train_run(parameters):
    for epoch in range(epochs):
        epoch_losses = []

        for X_batch, y_batch in tqdm(
            train_minibatches, desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            y_pred = model(X_batch)

            loss = cross_entropy(y_batch, y_pred)

            parameters = optimizer.step(loss, parameters)

            epoch_losses.append(loss.data)
            batch_loss_history.append(loss.data)

        epoch_loss_mean = float(np.mean(epoch_losses))
        mlflow.log_metric("train_loss", epoch_loss_mean, step=epoch)

        # scheduler.step(optimizer)
        mlflow.log_metric("learning_rate", optimizer.lr, step=epoch)
    return parameters


with mlflow.start_run() as run:
    mlflow.log_params(hyper_params)
    result1 = train_run(parameters)

    validation_accuracy = evaluate_model(model, test_minibatches)
    mlflow.log_metric("val accuracy", validation_accuracy)
    print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
