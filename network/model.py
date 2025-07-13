import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size) * 0.049
        self.b1 = np.zeros((hidden_size, 1))

        self.W2 = np.random.randn(output_size, hidden_size) * 0.049
        self.b2 = np.zeros((output_size, 1))


    def relu(self, Z):
        """
        ReLU activation function.
        
        Args:
            Z (array): Input data.
            
        Returns:
            array: Activated output.
        """
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        """
        Softmax activation function.
        
        Args:
            Z (array): Input data.
            
        Returns:
            array: Activated output.
        """
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X (array): Input data.
            
        Returns:
            array: Output of the network.
        """
        z1 = np.dot(self.W1, X) + self.b1
        self.a1 = self.relu(z1)
        
        z2 = np.dot(self.W2, self.a1) + self.b2
        self.y_hat = self.softmax(z2)
        
        return self.y_hat
    
    def backward(self, X, y):
        """
        Backward pass through the network.
        Args:
            X (array): Input data.
            y (array): True labels. 

        Returns:
            tuple: Gradients of W1, b1, W2, b2.
        """

        m = y.shape[1]
        dZ2 = self.y_hat - y
        self.dW2 = (1 / m) * np.dot(dZ2, self.a1.T)
        self.db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(self.a1 > 0))
        self.dW1 = (1 / m) * np.dot(dZ1, X.T)
        self.db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        return self.dW1, self.db1, self.dW2, self.db2
    
    def update_parameters(self, learning_rate):
        """
        Update the parameters of the network using gradient descent.
        
        Args:
            learning_rate (float): Learning rate for the update.
        """
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
    
