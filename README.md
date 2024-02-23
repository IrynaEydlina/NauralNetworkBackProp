# NeuralNetworkBackProp
A neural network that multiplies two floating-point numbers using the Backpropagation algorithm

Figure shows a neuron used as a basic building block in backpropagation networks. A set of inputs is provided, coming either from the outside or from the previous layer. Each is multiplied by the weight, and the products are summed up. This sum, denoted by NET, must be calculated for each neuron in the network. After calculating the NET value, it is modified using the activation function and the OUT signal is output.
![image](https://github.com/IrynaEydlina/NeuralNetworkBackProp/assets/24845420/1ba6dad6-f2d0-4cd7-8d30-afc75ef1330d)

A neural network consists of five layers. On the first there are two inputs for input data, on the last layer, there is one output for outputting the result.
Figure shows a sequential implementation of the training of an artificial neural network, the first two arguments are the input data, that is, the numbers to be multiplied, the third argument is the result that was calculated by the neural network, and the fourth argument is nothing but the correct answer. The absolute error is 5%
![image](https://github.com/IrynaEydlina/NeuralNetworkBackProp/assets/24845420/d0562fa4-7f62-4a9a-844f-a8929c6a7a62)

Graph of dependence of time on the number of processes:
![image](https://github.com/IrynaEydlina/NeuralNetworkBackProp/assets/24845420/e57f1f97-6143-41ab-8f6e-f5762e7842c3)
