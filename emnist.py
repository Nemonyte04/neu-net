import numpy as np
from sklearn.datasets import fetch_openml
from torchvision.datasets import EMNIST
import numpy as np

train = EMNIST(root="./data", split="letters",
              train=True, download=True)

test = EMNIST(root="./data", split="letters",
             train=False, download=True)

X_train = train.data.numpy().reshape(-1, 784) / 255.0
y_train = train.targets.numpy() - 1

X_test = test.data.numpy().reshape(-1, 784) / 255.0
y_test = test.targets.numpy() - 1

print(max(y_train))
