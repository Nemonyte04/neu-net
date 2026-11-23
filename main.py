from neuron import neuron
from neuron import layer
import numpy as np
import numpy.typing as npt
import math
from neuron import network
from sklearn.datasets import fetch_openml
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torchvision.datasets import EMNIST

# get the emnistt data

train = EMNIST(root="./data", split="letters",train=True, download=True)

test = EMNIST(root="./data", split="letters", train=False, download=True)

x_train = train.data.numpy().reshape(-1, 784) / 255.0
y_train = train.targets.numpy() - 1
y_train = np.eye(26,dtype = int)[y_train]
x_test = test.data.numpy().reshape(-1, 784) / 255.0
y_test = test.targets.numpy() - 1
y_test = np.eye(26,dtype = int)[y_test]




'''Uncomment if you want to view the image
print(X[0])
image_2d = x_train[0].reshape(28, 28)

# 2. Display the image
plt.imshow(image_2d, cmap="gray")
plt.show()
'''

#i still need to build an optimizer
print("-----Start training------")

nn = network(x_train[0])
epoch = 0
while epoch < 5:
    print("Epoch:", epoch)
    x_train, y_train = shuffle(x_train,y_train)
    for idx, data in enumerate(x_train):
        output, layer_1_output, layer_2_output = nn.feed_fwd(data)
        layer_1_gradient, layer_2_gradient, output_gradient = nn.backprop(output,layer_1_output, layer_2_output, y_train[idx],data)
        nn.update_params(grad1=layer_1_gradient, grad2=layer_2_gradient,outputgrad=output_gradient,lr=0.02)
    epoch +=1

predictions = []
for i,val in enumerate(x_test):
    prediction = np.argmax(nn.predict(val))
    true_value = np.argmax(y_test[i])
    predictions.append([prediction, true_value])


predictions = np.array(predictions)

accuracy = np.mean(predictions[:, 0] == predictions[:, 1])

print(f"Test accuracy: {accuracy:.2%}")