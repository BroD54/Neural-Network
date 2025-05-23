import numpy as np
from network.utils import cross_entropy_loss

def train(model, X_train, y_train, epochs, batch_size, learning_rate):
    m = X_train.shape[1] 

    for epoch in range(epochs):
        permutation = np.random.permutation(m)
    
        X_shuffled = X_train[:, permutation]
        y_shuffled = y_train[:, permutation]

        epoch_loss = 0
        num_batches = 0

        for start in range(0, m, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[:, start:end]
            y_batch = y_shuffled[:, start:end]
            
            y_hat = model.forward(X_batch)
            y_batch_labels = np.argmax(y_batch, axis=0)
            loss = cross_entropy_loss(y_hat, y_batch_labels)
            dW1, db1, dW2, db2 = model.backward(X_batch, y_batch)
            model.update_parameters(learning_rate)

            epoch_loss += loss
            num_batches += 1
            
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / num_batches:.4f}")

