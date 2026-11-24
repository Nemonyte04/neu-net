import numpy as np
import numpy.typing as npt

'''
neuron class needs weights for itself, regression on those weights, activation
'''
class neuron:
    def __init__(self, weights: npt.NDArray):
        self.weights = weights
    def regress(self, inputs):
        return np.sum(self.weights * inputs)
    def activate(self, inputs):
        #this is relu activation for each neuron, although i might not use this as i build the layer class
        output = max(0,self.regress(inputs))
        return output

    def set_weights(self, weights_val):
        self.weights = weights_val
        return 0

# layer class, i will jusst build the functions as i go, for now it needs number of nodes, weights, and neurons for each node
class layer:
    def __init__(self, nodes, inp_size) -> None:

        #he initialization, change the intialization here

        weights = np.random.randn(nodes, inp_size) * np.sqrt(2.0 / inp_size)
        weights[:, 0] = 0  

        self.nodes = nodes
        self.neurons = np.array([None]*nodes)
        for i in range(nodes):
            self.neurons[i] = neuron(weights[i])
        self.m = np.zeros((nodes,inp_size))
        self.v = np.zeros((nodes,inp_size))
    def regress(self,inp):

        inp = np.insert(inp,0,1)
        return np.asarray([n.regress(inp) for n in self.neurons])

    def output(self,inp,fnc):
        #add more activation functions here
        if fnc == "relu":
            return np.maximum(0, self.regress(inp))
        elif fnc == "sftmx":
            z = self.regress(inp)
            z = z - np.max(z)
            return np.exp(z) / np.sum(np.exp(z))
        else:
            return "Activation function not valid"

    #derivative of the relu layer, any other activation function would need the derivative here
    def relu_der(self,inp):
        inp = np.insert(inp,0,1)
        weighted = np.asarray([n.regress(inp) for n in self.neurons])
        return np.where(weighted > 0, 1, 0)

    def get_weights(self):
        return np.asarray([n.weights for n in self.neurons])

    def update_weights(self, updates):
        for i,z in enumerate(self.neurons):
            z.set_weights(updates[i])


'''
Initalize network here:
    ---- Change the network archietcture here, i am testing it on the emnist dataset so i have chosen my archictecture based on that
'''
class network:
    def __init__(self, inputs) -> None:
        self.input_layer = inputs
        self.layer_1 = layer(nodes= 128, inp_size=self.input_layer.shape[0]+1)
        self.layer_2 = layer(64,128+1)
        self.output_layer = layer(26, 64+1)
    
    def feed_fwd(self, input_data):
            layer_1_output = self.layer_1.output(input_data, "relu")
            layer_2_output = self.layer_2.output(layer_1_output, "relu")
            output = self.output_layer.output(layer_2_output, "sftmx")

            return output, layer_1_output, layer_2_output
    

    def backprop(self, output,layer_1_output, layer_2_output, real, input_data):
        error = output - real
        
        
        output_gradient = np.outer(error,np.insert(layer_2_output,0,1))


        layer_2_error = (self.output_layer.get_weights()[:, 1:].T @ error) * self.layer_2.relu_der(layer_1_output)
        layer_2_gradient = np.outer(layer_2_error,np.insert(layer_1_output,0,1))

        layer_1_error = (self.layer_2.get_weights()[:, 1:].T @ layer_2_error) * self.layer_1.relu_der(input_data)
       
        layer_1_gradient = np.outer(layer_1_error , np.insert(input_data,0,1))

        return layer_1_gradient, layer_2_gradient, output_gradient

    def update_params(self, grad1, grad2, outputgrad, run, lr = 0.02):

        m = 0.9 * self.output_layer.m + (1 - 0.9) * outputgrad
        v = 0.999 * self.output_layer.v + (1 - 0.999) * (outputgrad**2) 
        self.output_layer.m = m
        self.output_layer.v = v
        m_hat = m / (1 - 0.9**run)
        v_hat = v / (1 - 0.999**run)
        self.output_layer.update_weights(self.output_layer.get_weights()
                - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            )
        m_2 = 0.9 * self.layer_2.m + (1 - 0.9) * grad2
        v_2 = 0.999 * self.layer_2.v + (1 - 0.999) * (grad2**2)
        self.layer_2.m = m_2
        self.layer_2.v = v_2
        m_hat_2 = m_2 / (1-0.9**run)
        v_hat_2 = v_2 / (1-0.999**run)

        self.layer_2.update_weights(
                    self.layer_2.get_weights() - lr * m_hat_2 /( np.sqrt(v_hat_2)+ 1e-8)
                )
        
        self.layer_1.update_weights(
                            self.layer_1.get_weights() - lr * grad1
                        )

        return self.layer_1.get_weights(), self.layer_2.get_weights(), self.output_layer.get_weights()
    
    def predict(self, input_data):
            layer_1_output = self.layer_1.output(input_data, "relu")
            layer_2_output = self.layer_2.output(layer_1_output, "relu")
            output = self.output_layer.output(layer_2_output, "sftmx")

            return output
'''
Backprop formula for ReLu hidden layers and softmax activation layers, found them on a medium article so hopefully it checks out
Forward propagation:
# a^(l) = f(W^(l) @ a_tilde^(l-1))

Output-layer error (softmax + cross-entropy):
delta^(L) = y_pred - y_true

Hidden-layer error:
delta^(l) = (W_no_bias^(l+1).T @ delta^(l+1)) * f'(z^(l))

Weight gradient:
dL/dW^(l) = delta^(l) @ a_tilde^(l-1).T

Weight update (gradient descent):
W^(l) = W^(l) - eta * dL/dW^(l)

'''