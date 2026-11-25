# neu-net
Neural network only using numpy in python under a 100 lines of code
- Built this on my thanksgiving break train ride
- Cross entropy loss, relu activation, softmax output (all which can be changed in the layer class)
- I tested on the EMNIST (english letters dataset)

### Usage:
- Clone the repo, pip install -r requirements.txt, then run main.py. 
- Make changes to the train-tes pipeline on main.py, changes to the network on neuron.py
- One of the network implementations has Adam optimizer in the last 2 layers, while the other is simple SGD throughout

### Results:
- ~97% on MNIST numbers data
- ~89% on English letters data (combined 26 classes)