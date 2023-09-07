from numpy import random, array, dot, tanh

class NeuralNetwork:
    def __init__(self):
        random.seed(1)
        self.weight_matrix = 2*random.random((3,1))-1
    def tanh(self,x):
        return tanh(x)
    def tanh_derivative(self,x):
        return 1-tanh(x)**2
    def forward_propagation(self,inputs):
        return self.tanh(dot(inputs,self.weight_matrix))
    def train(self,train_inputs,train_outputs,num_train_iterations):
        for iteration in range(num_train_iterations):
            output = self.forward_propagation(train_inputs)
            error = train_outputs-output
            adjustment = dot(train_inputs.T,error*self.tanh_derivative(output))
            self.weight_matrix+=adjustment

if __name__=="__main__":

    neural_network = NeuralNetwork()
    print("Random Weights at start of training")
    print(neural_network.weight_matrix)
    train_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    train_outputs = array([[0,1,1,0]]).T

    neural_network.train(train_inputs, train_outputs, 10000)
    print("New weights after training")
    print(neural_network.weight_matrix)

    print("Testing Network on new examples")
    print(neural_network.forward_propagation(array([1,0,0])))
