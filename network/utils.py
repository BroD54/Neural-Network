import numpy as np
import matplotlib.pyplot as plt

def cross_entropy_loss(y_hat, y_true):
    """
    Compute the cross-entropy loss.
    
    Args:
        y_hat (array): Predicted output.
        y_true (array): True labels.
        
    Returns:
        float: Cross-entropy loss.
    """
    m = y_true.shape[0]
    correct_probs = y_hat[y_true, np.arange(m)]
    loss = -np.sum(np.log(correct_probs)) / m

    return loss

def accuracy(y_hat, y_true):
    """
    Compute the accuracy of the model.
    
    Args:
        y_hat (array): Predicted output.
        y_true (array): True labels.
        
    Returns:
        float: Accuracy of the model.
    """
    predictions = np.argmax(y_hat, axis=0)
    labels = np.argmax(y_true, axis=0)

    return np.mean(predictions == labels)

def visualize_predictions(model, X_test, y_test, num_images=10):
    """
    Visualizes images with their predicted and true labels.
    
    Args:
        model: Your trained model.
        X_test (np.array): Test images, shape (features, samples).
        y_test (np.array): True labels in one-hot format, shape (classes, samples).
        num_images (int): Number of images to display.
    """
    for i in range(num_images):
        img = X_test[:, i].reshape(28, 28)
        true_label = np.argmax(y_test[:, i])
        
        y_hat = model.forward(X_test[:, i:i+1])
        predicted_label = np.argmax(y_hat)
        
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_label}, Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()
