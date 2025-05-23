from data.data_loader import load_mnist, one_hot_encode
from network.model import NeuralNetwork
from network.train import train
from network.utils import accuracy, visualize_predictions

def main():
    (x_train, y_train), (x_test, y_test) = load_mnist()

    y_train_oh = one_hot_encode(y_train)
    y_test_oh = one_hot_encode(y_test)
    
    input_size = 784
    hidden_size = 64
    output_size = 10
    
    model = NeuralNetwork(input_size, hidden_size, output_size)
    
    epochs = 10
    batch_size = 64
    learning_rate = 0.01
    
    train(model, x_train, y_train_oh, epochs, batch_size, learning_rate)
    
    y_pred = model.forward(x_test)
    test_acc = accuracy(y_pred, y_test_oh)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    visualize_predictions(model, x_test, y_test_oh, num_images=10)
    
if __name__ == "__main__":
    main()