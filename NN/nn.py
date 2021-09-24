from re import T
from typing import final
import numpy as np
import scipy.special
from keras.datasets import mnist
from matplotlib import pyplot as plt
from tqdm import tqdm

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = 0.2

        if(False):
            with open('weights0.npy', 'rb') as f:
                self.input_weights = np.load(f)
            with open('weights1.npy', 'rb') as f:
                self.hidden_weights = np.load(f)

        else:
            self.input_weights = np.random.rand(self.hidden_nodes, self.input_nodes)-0.5
            self.hidden_weights = np.random.rand(self.output_nodes, self.hidden_nodes)-0.5
        
        self.activation_function = lambda x : scipy.special.expit(x)
        pass

    # Train Neural Network
    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        #1. X(h) = I * W(i-h)
        hidden_inputs = np.dot(self.input_weights, inputs)
        
        #2. O(h) = sigmoid(X(h))
        hidden_outputs = self.activation_function(hidden_inputs)

        #3. X(o) = O(h) * W(h-o)
        final_inputs = np.dot(self.hidden_weights, hidden_outputs)

        #4. O = sigmoid(X(o))
        final_outputs = self.activation_function(final_inputs)

        #5. Error
        output_errors = targets - final_outputs

        #6. Errors to hidden
        hidden_errors = np.dot(self.hidden_weights.T, output_errors)

        #7. Adjust weights
        #self.weights[1] += self.learning_rate * output_errors * final_outputs * (1 - final_outputs) * hidden_outputs.T
        self.hidden_weights += self.learning_rate * np.dot((output_errors*final_outputs*(1-final_outputs)),np.transpose(hidden_outputs))

        self.input_weights += self.learning_rate * np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),np.transpose(inputs))


    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2).T

        #1. X(h) = I * W(i-h)
        hidden_inputs = np.dot(self.input_weights, inputs)
        
        #2. O(h) = sigmoid(X(h))
        hidden_outputs = self.activation_function(hidden_inputs)

        #3. X(o) = O(h) * W(h-o)
        final_inputs = np.dot(self.hidden_weights, hidden_outputs)

        #4. O = sigmoid(X(o))
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    def get_result(self, final_outputs):
        return np.argmax(final_outputs)

    def save(self):
        with open('input.npy', 'wb') as f:
            np.save(f, self.input_weights, allow_pickle=True)
        with open('hidden.npy', 'wb') as f:
            np.save(f, self.hidden_weights, allow_pickle=True)



def train(train_data, train_labels):
    for row in tqdm(range(60000)):
        data = extremify(train_data[row])
        labels = train_labels[row].flatten()
        correct = np.zeros(output_size)
        correct[labels[0]] = 1
        nn.train(data, correct)

def test(test_data, test_labels):
    right = 0
    for row in tqdm(range(10000)):
        data = extremify(test_data[row])
        labels = test_labels[row].flatten()
        correct = np.zeros(output_size)
        correct[labels[0]] = 1

        result = nn.get_result(nn.query(data))

        #print(result)
        #show(data)

        if(result==labels[0]):
            right += 1
    print("Accuracy: ", right/10000)

def extremify(img):
    img = img.flatten()
    for pixel in range(len(img)):
        if(img[pixel]>127):
            img[pixel] = 255
        else:
            img[pixel] = 0
    return img

def show(img):
    img = np.array(img, dtype='float')
    pixels = img.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

if __name__ == '__main__':
    input_size = 784
    hidden_size = 100
    output_size = 10

    nn = NeuralNetwork(input_size, hidden_size, output_size)
    
    print("----------------")
    print('Loading training data...')
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    print('Done!')
    print("----------------")

    print("Training...")
    train(train_data, train_labels)
    print('Done!')
    print("----------------")

    print("Testing...")
    test(test_data, test_labels)

    '''b = time.time()
    for row in tqdm(range(10000)):
        extremify(test_data[row])
    e = time.time()

    print("Time: ", e-b)'''


    nn.save()