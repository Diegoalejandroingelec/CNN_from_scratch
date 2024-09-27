from tqdm import tqdm  # Import tqdm for displaying a progress bar
from utils import visualize_tensor
# NeuralNetwork class definition
class NeuralNetwork():
    def __init__(self, model, loss_f, loss_prime, lr):
        """
        Initialize the neural network with the given model, loss functions, and learning rate.
        
        Args:
            model: List of layers in the neural network.
            loss_f: Loss function for calculating the error.
            loss_prime: Derivative of the loss function for backpropagation.
            lr: Learning rate for gradient descent.
        """
        self.model = model
        self.loss_prime = loss_prime
        self.loss_f = loss_f
        self.lr = lr

    def predict(self, x, visualize_features=False):
        """
        Perform a forward pass through the network to make a prediction.
        
        Args:
            x: Input data.
        
        Returns:
            output: The output after passing through all layers of the network.
        """
        original_input = x.copy()
        output = x
        for layer in self.model:
            output = layer.forward(output)  # Pass the data through each layer
            
            if(type(layer).__name__ == 'Sigmoid' and  output.shape ==(6,26,26) and visualize_features):
                visualize_tensor(original_input, output)
            

        return output

    def back_propagation(self, prediction, ground_truth):
        """
        Perform backpropagation to update the weights and biases based on the prediction and ground truth.
        
        Args:
            prediction: The predicted output from the forward pass.
            ground_truth: The actual target output.
        
        Returns:
            error: The calculated error between the prediction and ground truth.
        """
        # Calculate the error using the loss function
        error = self.loss_f(prediction, ground_truth)
        
        # Calculate the gradient of the loss with respect to the prediction
        gradient = self.loss_prime(prediction, ground_truth)
        
        # Backpropagate the gradient through each layer in reverse order
        for layer in reversed(self.model):
            gradient = layer.backward(gradient, self.lr)
        
        return error

    def get_network_info(self):
        """
        Retrieve detailed information about the network, such as weights, biases, layer names, 
        and input/output shapes for each layer.
        
        Returns:
            network_w: List of weights for each layer.
            network_B: List of biases for each layer.
            layers_name: List of layer names.
            layers_shapes: Dictionary containing input and output shapes for each layer.
        """
        network_w = []
        network_B = []
        layers_name = []
        layers_shapes = {'layers_input_shape': [], 'layers_output_shape': []}

        for layer in self.model:
            # Check if layer has weights 'W' and biases 'B'
            if hasattr(layer, 'W'):
                network_w.append(layer.W)
                network_B.append(layer.B)
                layers_name.append(type(layer).__name__)
                layers_shapes['layers_input_shape'].append(layer.input_shape)
                layers_shapes['layers_output_shape'].append(layer.output_shape)
            # Check if layer has kernels 'K' and biases 'B'
            elif hasattr(layer, 'K'):
                network_w.append(layer.K)
                network_B.append(layer.B)
                layers_name.append(type(layer).__name__)
                layers_shapes['layers_input_shape'].append(layer.input_shape)
                layers_shapes['layers_output_shape'].append(layer.output_shape)
            # For layers without weights/biases
            else:
                if hasattr(layer, 'input_shape') and hasattr(layer, 'output_shape'):
                    layers_shapes['layers_input_shape'].append(layer.input_shape)
                    layers_shapes['layers_output_shape'].append(layer.output_shape)
                else:
                    layers_shapes['layers_input_shape'].append(layers_shapes['layers_output_shape'][-1])
                    layers_shapes['layers_output_shape'].append(layers_shapes['layers_output_shape'][-1])

                network_w.append(None)
                network_B.append(None)
                layers_name.append(type(layer).__name__)

        return network_w, network_B, layers_name, layers_shapes

    def summary(self):
        """
        Display a summary of the network's architecture, including the input/output shapes 
        and the number of parameters for each layer.
        """
        network_w, network_B, layers_name, layers_shapes = self.get_network_info()
        print('')
        print('######### NETWORK SUMMARY #########')
        print('')
        
        # Print the summary for each layer
        for w, b, layer_name, layer_input_shape, layer_output_shape in zip(network_w, network_B, layers_name, layers_shapes['layers_input_shape'], layers_shapes['layers_output_shape']):
            if w is not None and b is not None:
                print('')
                print(f'{layer_name}')
                print(f'Input shape: {layer_input_shape}')
                print(f'Output shape: {layer_output_shape}')
                print(f'Weights: shape--> {w.shape}  #Parameters--> {w.size}')
                print(f'Biases:  shape--> {b.shape}  #Parameters--> {b.size}')
                print(f'The total number of parameters in this layer is {w.size + b.size}')
                print('')
            else:
                print('')
                print(f'{layer_name}')
                print(f'Input shape: {layer_input_shape}')
                print(f'Output shape: {layer_output_shape}')
                print('This layer does not have trainable parameters')
                print('')

    def fit(self, X, Y, X_test, Y_test, epochs, verbose):
        """
        Train the neural network on the training data and evaluate it on the test data.
        
        Args:
            X: Training data input.
            Y: Training data labels.
            X_test: Test data input.
            Y_test: Test data labels.
            epochs: Number of epochs to train.
            verbose: If 1, print loss information for each epoch.
        
        Returns:
            training_errors: List of average training errors for each epoch.
            testing_errors: List of average testing errors for each epoch.
        """
        training_errors = []
        testing_errors = []

        for i in range(epochs):
            error = 0

            # Use tqdm for displaying a progress bar for each epoch
            for x, y in tqdm(zip(X, Y), total=len(X), desc=f"Epoch {i + 1}/{epochs}"):
                prediction = self.predict(x)  # Forward pass
                error += self.back_propagation(prediction, y)  # Backpropagation and error accumulation

            # Calculate and store average training error for the epoch
            training_errors.append(error / len(X))
            
            if verbose == 1:
                print(f"Loss value ----> {error / len(X):.6f} for epoch: {i + 1}")

            # Evaluate the network on the test set
            testing_error = 0
            for x_1, y_1 in tqdm(zip(X_test, Y_test), total=len(X_test), desc=f"Testing..."):
                testing_prediction = self.predict(x_1)  # Forward pass for test data
                testing_error += self.loss_f(testing_prediction, y_1)  # Compute test error

            # Store average testing error for the epoch
            testing_errors.append(testing_error / len(X_test))

        return training_errors, testing_errors