from tensorflow.keras.datasets import mnist
import numpy as np

def load_mnist():
    """
    Load the MNIST dataset and return the training and test data.
    
    Returns:
        tuple: A tuple containing the training and test data.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train.reshape(x_train.shape[0], -1).T
    x_test = x_test.reshape(x_test.shape[0], -1).T
    
    return (x_train, y_train), (x_test, y_test)


def one_hot_encode(y, num_classes=10):
    """
    One-hot encode the labels.
    
    Args:
        y (array): The labels to be one-hot encoded.
        num_classes (int): The number of classes.
        
    Returns:
        array: One-hot encoded labels.
    """
    return np.eye(num_classes)[y].T




# def test_load_mnist():
#     (x_train, y_train), (x_test, y_test) = load_mnist()

#     print("x_train shape:", x_train.shape)  # Should be (784, 60000)
#     print("y_train shape:", y_train.shape)  # Should be (60000,)
#     print("x_test shape:", x_test.shape)    # Should be (784, 10000)
#     print("y_test shape:", y_test.shape)    # Should be (10000,)

#     assert x_train.shape == (784, 60000)
#     assert x_test.shape == (784, 10000)
#     assert y_train.shape == (60000,)
#     assert y_test.shape == (10000,)
#     assert (x_train >= 0).all() and (x_train <= 1).all(), "x_train not normalized"

#     print("[OK] load_mnist passed!")

# def test_one_hot_encode():
#     import numpy as np
#     y = np.array([0, 3, 5])
#     encoded = one_hot_encode(y)

#     print("One-hot shape:", encoded.shape)  # Should be (10, 3)
#     print(encoded)

#     assert encoded.shape == (10, 3)
#     assert (encoded.sum(axis=0) == 1).all(), "Each column should sum to 1"

#     print("[OK] one_hot_encode passed!")

# if __name__ == "__main__":
#     test_load_mnist()
#     test_one_hot_encode()