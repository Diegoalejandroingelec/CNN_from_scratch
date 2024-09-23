from tqdm import tqdm

class NeuralNetwork():
    def __init__(self,model,loss_f,loss_prime,lr):
        self.model = model
        self.loss_prime = loss_prime
        self.loss_f =loss_f
        self.lr = lr

    def predict(self,x):
        output = x
        for layer in self.model:
            output=layer.forward(output)
        return output
    
    def back_propagation(self,prediction,ground_truth):
        
        error=self.loss_f(prediction,ground_truth)

        gradient = self.loss_prime(prediction,ground_truth)
        
        for layer in reversed(self.model):
            gradient = layer.backward(gradient,self.lr)

        return error
    
    def get_network_info(self):
        network_w=[]
        network_B=[]
        layers_name = []
        layers_shapes = {'layers_input_shape':[],'layers_output_shape':[]}
        for layer in self.model:
            if(hasattr(layer, 'W')):
                network_w.append(layer.W)
                network_B.append(layer.B)
                layers_name.append(type(layer).__name__)
                layers_shapes['layers_input_shape'].append(layer.input_shape)
                layers_shapes['layers_output_shape'].append(layer.output_shape)
            elif(hasattr(layer, 'K')):
                network_w.append(layer.K)
                network_B.append(layer.B)
                layers_name.append(type(layer).__name__)
                layers_shapes['layers_input_shape'].append(layer.input_shape)
                layers_shapes['layers_output_shape'].append(layer.output_shape)
            else:
                if(hasattr(layer, 'input_shape') and hasattr(layer, 'output_shape')):
                    layers_shapes['layers_input_shape'].append(layer.input_shape)
                    layers_shapes['layers_output_shape'].append(layer.output_shape)
                else:
                    layers_shapes['layers_input_shape'].append(layers_shapes['layers_output_shape'][-1])
                    layers_shapes['layers_output_shape'].append(layers_shapes['layers_output_shape'][-1])

                network_w.append(None)
                network_B.append(None)
                layers_name.append(type(layer).__name__)

           
        return network_w, network_B,layers_name,layers_shapes
    
    def summary(self):

        network_w, network_B,layers_name,layers_shapes = self.get_network_info()
        print('')
        print('######### NETWORK SUMMARY #########')
        print('')
        for w,b,layer_name,layer_input_shape,layer_output_shape in zip(network_w, network_B,layers_name,layers_shapes['layers_input_shape'],layers_shapes['layers_output_shape']):
            if(w is not None and b is not None):
                print('')
                print(f'{layer_name}')
                print(f'Input shape: {layer_input_shape}')
                print(f'Output shape: {layer_output_shape}')
                print(f'Weights: shape--> {w.shape}  #Parameters--> {w.size}')
                print(f'Biases:  shape--> {b.shape}  #Parameters--> {b.size}')
                print(f'The total number of parameter in this layer is {w.size+b.size}')
                print('')
            else:
                print('')
                print(f'{layer_name}')
                print(f'Input shape: {layer_input_shape}')
                print(f'Output shape: {layer_output_shape}')
                print('This layer does not have trainable parameters')
                print('')

    
    def fit(self, X,Y,X_test,Y_test, epochs, verbose):
       
        training_errors = []
        testing_errors = []
        for i in range(epochs):
            error = 0
            
            # Using tqdm for progress bar
            for x, y in tqdm(zip(X, Y), total=len(X), desc=f"Epoch {i + 1}/{epochs}"):
                prediction = self.predict(x)
                error += self.back_propagation(prediction, y)
            
           
            training_errors.append(error/len(X))
            if verbose == 1:
                print(f"Loss value ----> {error / len(X):.6f} for epoch: {i + 1}")

            testing_error = 0
            for x_1, y_1 in tqdm(zip(X_test, Y_test), total=len(X_test), desc=f"Testing..."):
                testing_prediction = self.predict(x_1)
                testing_error+=self.loss_f(testing_prediction,y_1)
            
            testing_errors.append(testing_error/len(X_test))


        return training_errors,testing_errors
