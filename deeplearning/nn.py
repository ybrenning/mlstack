import numpy as np


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1. - sigmoid(z))


class NN:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.w = [np.random.randn(y, x)
                  for x, y in zip(layers[:-1], layers[1:])]
        self.b = [np.random.randn(y, 1) for y in layers[1:]]

    def forward(self, a):
        for w, b in zip(self.w, self.b):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def batch_train(self, X_train, y_train, epochs, batch_size, lr=0.01):
        for i in range(0, epochs):
            x_batches = [
                X_train[k:k+batch_size]
                for k in range(0, len(X_train), batch_size)
            ]
            y_batches = [
                y_train[k:k+batch_size]
                for k in range(0, len(y_train), batch_size)
            ]

            for xb, yb in zip(x_batches, y_batches):
                self.train(xb, yb, lr)

            print(f"Epoch {i} completed.")

    def train(self, X_batch, y_batch, lr):
        assert (len(X_batch) == len(y_batch))

        nabla_w = [np.zeros(w.shape) for w in self.w]
        nabla_b = [np.zeros(b.shape) for b in self.b]

        for X, y in zip(X_batch, y_batch):
            delta_nabla_w, delta_nabla_b = self.backprop(X, y)
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        self.w = [w-(lr/len(X_batch))*nw for w, nw in zip(self.w, nabla_w)]
        self.b = [b-(lr/len(X_batch))*nb for b, nb in zip(self.b, nabla_b)]

    def backprop(self, X, y):
        nabla_w = [np.zeros(w.shape) for w in self.w]
        nabla_b = [np.zeros(b.shape) for b in self.b]

        activation = X
        activations = [X]
        zs = []
        for w, b in zip(self.w, self.b):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            activations.append(activation)

            delta = self.cost_prime(activations[-1], y) * sigmoid_prime(zs[-1])
            nabla_w[-1] = np.dot(delta, activations[-2].T)
            nabla_b[-1] = delta

            for layer in range(2, self.num_layers):
                z = zs[-layer]
                sp = sigmoid_prime(z)
                delta = np.dot(self.w[-layer+1].T, delta) * sp
                nabla_w[-layer] = delta
                nabla_b[-layer] = np.dot(delta, activations[-layer+1].T)

            return (nabla_w, nabla_b)

        def cost_prime(self, output_activations, y):
            return output_activations - y


if __name__ == "__main__":
    ...
