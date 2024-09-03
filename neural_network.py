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
    
    def get_weights(self):
        network_w=[]
        network_B=[]
        for layer in self.model:
           
           try:
            network_w.append(layer.W)
            network_B.append(layer.B)
           except:
            network_w.append(None)
            network_B.append(None)
           
        return network_w, network_B
    
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
