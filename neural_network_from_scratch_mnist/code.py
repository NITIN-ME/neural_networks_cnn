import numpy as np
import mnist

class FullyConnectedLayer(object):

    def __init__(self, num_inputs, layer_size, activation_function, derivated_activation_function=None):
        super().__init__()
        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_function = activation_function
        self.derivated_activation_function = derivated_activation_function
        self.x, self.y = None, None
        self.dL_dW, self.dL_db = None, None

    def forward(self, x):
        z = np.dot(x, self.W) + self.b
        self.y = self.activation_function(z)
        self.x = x
        return self.y


    def backward(self, dL_dy):
        dy_dz = self.derivated_activation_function(self.y)
        dL_dz = (dL_dy * dy_dz)
        dz_dw = self.x.T
        dz_dx = self.W.T
        dz_db = np.ones(dL_dy.shape[0])

        self.dL_dW = np.dot(dz_dw, dL_dz)
        self.dL_db = np.dot(dz_db, dL_dz)

        dL_dx = np.dot(dL_dz, dz_dx)
        return dL_dx

    def optimize(self, epsilon):
        self.W -= epsilon * self.dL_dW
        self.b -= epsilon * self.dL_db


class Neuron(object):
    
    def __init__(self, num_inputs, activation_function):
        super().__init__()
        self.W = np.random.uniform(size=num_inputs, low=-1., high=1.)
        self.b = np.random.uniform(size=1, low=-1., high=1.)
        self.activation_function = activation_function

    def forward(self, x):
        z = np.dot(x, self.W) + self.b
        return self.activation_function(z)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivated_sigmoid(y):
    return y * (1 - y)


def loss_L2(pred, target):
    return np.sum(np.square(pred - target)) / pred.shape[0]


def derivated_loss_L2(pred, target):
    return 2 * (pred - target)


def cross_entropy(pred, target):
    return -np.mean(np.multiply(np.log(pred), target) + np.multiply(np.log(1 - pred), (1 - target)))


def derivated_cross_entropy(pred, target):
    return (pred - target) / (pred * (1 - pred))


class SimpleNetwork(object):
    def __init__(self, num_inputs, num_outputs, hidden_layers_sizes=(64, 32),
                activation_function=sigmoid, derivated_activation_function=derivated_sigmoid,
                loss_function=loss_L2, derivated_loss_function=derivated_loss_L2):
        super().__init__()
        layer_sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
        self.layers = [
            FullyConnectedLayer(layer_sizes[i], layer_sizes[i + 1], activation_function, derivated_activation_function)
            for i in range(len(layer_sizes) - 1)
            ]

        self.loss_function = loss_function
        self.derivated_loss_function = derivated_loss_function

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x):
        estimations = self.forward(x)
        best_class = np.argmax(estimations)
        return best_class

    def backward(self, dL_dy):
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)
        return dL_dy

    def optimize(self, epsilon):
        for layer in self.layers:
            layer.optimize(epsilon)

    def evaluate_accuracy(self, X_val, y_val):
        num_corrects = 0
        for i in range(len(X_val)):
            pred_class = self.predict(X_val[i])
            if pred_class == y_val[i]:
                num_corrects += 1
        return num_corrects / len(X_val)

    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, num_epochs=5, learning_rate=1e-3):
        num_batches_per_epoch = len(X_train) // batch_size
        do_validation = X_val is not None and y_val is not None
        losses, accuracies = [], []
        for i in range(num_epochs):
            epoch_loss = 0
            for b in range(num_batches_per_epoch):
                batch_index_begin = b * batch_size
                batch_index_end = batch_index_begin + batch_size
                x = X_train[batch_index_begin: batch_index_end]
                targets = y_train[batch_index_begin: batch_index_end]
                predictions = y = self.forward(x)
                L = self.loss_function(predictions, targets)
                dL_dy = self.derivated_loss_function(predictions, targets)
                self.backward(dL_dy)
                self.optimize(learning_rate)
                epoch_loss += L

            epoch_loss /= num_batches_per_epoch
            losses.append(epoch_loss)
            if do_validation:
                accuracy = self.evaluate_accuracy(X_val, y_val)
                accuracies.append(accuracy)
            else:
                accuracy = np.NaN
            print("Epoch {:4d}: training loss = {:.6f} | val accuracy = {:.2f}%".format(i, epoch_loss, accuracy * 100))
        return losses, accuracies



np.random.seed(42)

X_train, y_train = mnist.train_images(), mnist.train_labels()
X_test,  y_test  = mnist.test_images(), mnist.test_labels()
num_classes = 10
X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)
y_train = np.eye(num_classes)[y_train]
mnist_classifier = SimpleNetwork(X_train.shape[1], num_classes, [64, 32])
losses, accuracies = mnist_classifier.train(X_train, y_train, X_test, y_test, batch_size=30, num_epochs=500)



