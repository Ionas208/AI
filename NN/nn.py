import numpy as np
import cupy as cp
import scipy.special
from matplotlib import pyplot as plt
from tqdm import tqdm
from mnist import MNIST
from numba import njit

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, load = True):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = 0.001

        if(load):
            with open('input_weights.npy', 'rb') as f:
                self.input_weights = cp.load(f)
            with open('hidden_weights.npy', 'rb') as f:
                self.hidden_weights = cp.load(f)

        else:
            self.input_weights = cp.random.rand(self.hidden_nodes, self.input_nodes)-0.5
            self.hidden_weights = cp.random.rand(self.output_nodes, self.hidden_nodes)-0.5
        
        self.activation_function = lambda x : self.sigmoid(x)
        pass

    def sigmoid(self, x):
        return 1/(1+cp.exp(-x))

    # Train Neural Network
    def train(self, inputs, targets):
        inputs = cp.array(inputs, ndmin=2).T
        targets = cp.array(targets, ndmin=2).T

        #1. X(h) = I * W(i-h)
        hidden_inputs = cp.dot(self.input_weights, inputs)
        
        #2. O(h) = sigmoid(X(h))
        hidden_outputs = self.activation_function(hidden_inputs)

        #3. X(o) = O(h) * W(h-o)
        final_inputs = cp.dot(self.hidden_weights, hidden_outputs)

        #4. O = sigmoid(X(o))
        final_outputs = self.activation_function(final_inputs)

        #5. Error
        output_errors = targets - final_outputs

        #6. Errors to hidden
        hidden_errors = cp.dot(self.hidden_weights.T, output_errors)

        #7. Adjust weights
        self.hidden_weights += self.learning_rate * cp.dot((output_errors*final_outputs*(1-final_outputs)),cp.transpose(hidden_outputs))

        self.input_weights += self.learning_rate * cp.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),cp.transpose(inputs))


    def query(self, inputs):
        inputs = cp.array(inputs, ndmin=2).T

        #1. X(h) = I * W(i-h)
        hidden_inputs = cp.dot(self.input_weights, inputs)
        
        #2. O(h) = sigmoid(X(h))
        hidden_outputs = self.activation_function(hidden_inputs)

        #3. X(o) = O(h) * W(h-o)
        final_inputs = cp.dot(self.hidden_weights, hidden_outputs)

        #4. O = sigmoid(X(o))
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    def save(self):
        with open('input_weights.npy', 'wb') as f:
            cp.save(f, self.input_weights, allow_pickle=True)
        with open('hidden_weights.npy', 'wb') as f:
            cp.save(f, self.hidden_weights, allow_pickle=True)



@njit
def extremify(img):
    img = img.flatten()
    for pixel in range(len(img)):
        if(img[pixel]>127):
            img[pixel] = 255
        else:
            img[pixel] = 0
    return img

class NNTrainer:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
        pass


    def get_result(self, final_outputs):
        return cp.argmax(final_outputs)

    def load(self, doExtremify = False):
        print("----------------")
        print('Loading training data...')
        mndata = MNIST('samples')
        mndata.gz = True
        train_data, train_labels = mndata.load_training()
        test_data, test_labels = mndata.load_testing()

        train_data = cp.array(train_data)
        train_labels = cp.array(train_labels)
        test_data = cp.array(test_data)
        test_labels = cp.array(test_labels)

        if(doExtremify):
            print("Extremifying...")
            for i in tqdm(range(len(train_data))):
                train_data[i] = extremify(train_data[i])
            for i in tqdm(range(len(test_data))):
                test_data[i] = extremify(test_data[i])

        self.train_data = cp.reshape(train_data, (60000, 28, 28))
        self.train_labels = cp.reshape(train_labels, (60000, 1))
        self.test_data = cp.reshape(test_data, (10000, 28, 28))
        self.test_labels = cp.reshape(test_labels, (10000, 1))

        train_mean_px = self.train_data.mean().astype(np.float32)
        train_std_px = self.train_data.std().astype(np.float32)
        self.train_data = (self.train_data - train_mean_px)/(train_std_px)

        test_mean_px = self.test_data.mean().astype(np.float32)
        test_std_px = self.test_data.std().astype(np.float32)
        self.test_data = (self.test_data - test_mean_px)/(test_std_px)



        #self.show(np.asarray(self.train_data[0]))
        print('Done!')
        print("----------------")
        pass

    def train(self, batch_size):
        self.test()
        print("Training...")
        for batch in range(batch_size):
            print('Batch #'+str(batch))
            for row in tqdm(range(60000)):
                data = self.train_data[row].flatten()
                labels = self.train_labels[row].flatten()
                correct = cp.zeros(self.output_nodes)
                correct[labels[0]] = 1
                self.nn.train(data, correct)
            self.test()
        self.nn.save()
        print('Done!')
        print("----------------")
        pass

    def test(self):
        print("Testing...")
        right = 0
        for row in tqdm(range(10000)):
            data = self.test_data[row].flatten()
            labels = self.test_labels[row].flatten()
            correct = cp.zeros(self.output_nodes)
            correct[labels[0]] = 1

            result = self.get_result(self.nn.query(data))

            if(result==labels[0]):
                right += 1

        print('Done!')
        print('Accuracy: '+str(right/10000))
        print("----------------")
        return right/10000
    
    def show(self, img):
        pixels = img.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()


if __name__ == '__main__':
    cp.cuda.Device(0).use()
    input_size = 784
    hidden_size = 524
    output_size = 10
    NNT = NNTrainer(input_size, hidden_size, output_size)
    NNT.load()
    NNT.train(5)