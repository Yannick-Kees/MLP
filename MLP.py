import numpy as np
import sys

np.random.seed(100)

def report_progress(current, total, error):
    # Prints out, how far the training process is

    # Parameters:
    #     current:    where we are right now
    #     total:      how much to go
    #     error:      Current Error, i.e. evaluation of the loss functional
   
   sys.stdout.write('\rProgress: {:.2%}, Current Error: {:}'.format(float(current)/total, error))
   
   if current==total:
      sys.stdout.write('\n')
      
   sys.stdout.flush()

class Layer:
    # Initialize an emtpy layer with no weights
    def __init__(self, number_of_neurons: int, activation_function, inputs=None, last=False):

        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function
        self.inputs = inputs
        self.shape = None
        self.data_shape = None
        self.weights = None
        self.bias = None
        self.__set_weights()

    def process(self, inputs):
        # Feed foward through layer
        self.inputs = inputs
        self.__set_weights()
        out = inputs @ self.weights + self.bias
        out = self.activation_function(out)
        return out

    def __set_weights(self):
        # Initialize weights
        if self.weights is None and self.inputs is not None:
            self.data_shape = self.inputs.shape
            self.shape = (self.data_shape[-1], self.number_of_neurons)
            self.weights = np.random.random(self.shape) * 4.0 - 2.0 
            self.bias = np.random.random(self.number_of_neurons) * 4.0 - 2.0 

 
class Model:
    def __init__(self):
        self.layers = np.array([], dtype=Layer)
        self.inputs = None
        self.targets = None
        self.outputs = []
        self.learning_rate = None
        self.epochs = None
        self.optimizer = None

    def add_layer(self, layer: Layer):
        # Add layer to MLP
        self.layers = np.append(self.layers, layer)

    def train(self, inputs, targets, learning_rate, epochs):
        # Train network
        self.__setup__(inputs, targets, learning_rate, epochs)
        errors = []
        
        for epoch in range(epochs):
            
            self.evaluate(inputs)
            error = abs(np.ravel(self.targets - self.outputs[-1])).mean()
            errors.append(error)
            report_progress(epoch, epochs-1,error)
            
            self.gradient_descend()
            

    def evaluate(self, inputs):
        # Feed forward through network
        self.outputs = []
        self.outputs.append(inputs)
        
        for i in range(len(self.layers)):
    
            inputs = self.layers[i].process(inputs)
            self.outputs.append(inputs)

        return inputs
    
    def hidden_states(self, inputs):
            return self.layers[0].process(inputs)

    def gradient_descend(self):
        # Optimizer
        i = self.layers.size
        predicted_output = self.outputs[-1]
        error = self.targets - predicted_output
        error = error * self.learning_rate
        
        for layer, predicted_output in zip(reversed(self.layers), reversed(self.outputs)):
         
            delta = error * layer.activation_function(predicted_output, derivative=True)
            error = delta @ layer.weights.T
            
            if i > 0:
                previous_output = self.outputs[i - 1]
                dw = previous_output.T @ delta
                db  = delta.mean()

                layer.weights += dw
                layer.bias += db
                
            i -= 1

    def __setup__(self, inputs, targets, learning_rate, epochs):
        self.inputs = inputs
        self.targets = targets
        self.learning_rate = learning_rate
        self.epochs = epochs


    
# Activation functions   
    
def identity(x, derivative=False):
    if derivative:
        return 1.0 
    else:
        return x
      
def tanh(x, derivative=False):
    if derivative:

        return 1.0 - np.tanh(x)**2
    else:
        return np.tanh(x)

def sgn(x, derivative=False):
    if derivative:

        return 1.0
    else:
        return 0.5 * np.sign(x) + .5
    

def create_network(layers, activation_function):
    # Create Model of MLP
    #   layers:                 Array containing layer sizes
    #   activation_function:    Activation in all layers
    mlp = Model()
    for l in layers[1:]:
        layer = Layer(l, activation_function)
        mlp.add_layer(layer)
    return mlp
    
