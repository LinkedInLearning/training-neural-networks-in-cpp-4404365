#include "MLP.h"

double frand(){
	return (2.0*(double)rand() / RAND_MAX) - 1.0;
}


// Return a new Perceptron object with the specified number of inputs (+1 for the bias).
Perceptron::Perceptron(size_t inputs, double bias){
	this->bias = bias;
	weights.resize(inputs+1);
	generate(weights.begin(),weights.end(),frand);
}

// Run the perceptron. x is a vector with the input values.
double Perceptron::run(std::vector<double> x){
	x.push_back(bias);
	double sum = inner_product(x.begin(),x.end(),weights.begin(),(double)0.0);
	return sigmoid(sum);
}

// Set the weights. w_init is a vector with the weights.
void Perceptron::set_weights(std::vector<double> w_init){
	weights = w_init;
}

// Evaluate the sigmoid function for the floating point input x.
double Perceptron::sigmoid(double x){
	return 1.0/(1.0 + exp(-x));
}


// Return a new MultiLayerPerceptron object with the specified parameters.
MultiLayerPerceptron::MultiLayerPerceptron(std::vector<size_t> layers, double bias, double eta) {
    this->layers = layers;
    this->bias = bias;
    this->eta = eta;

    for (size_t i = 0; i < layers.size(); i++){
        values.push_back(std::vector<double>(layers[i],0.0));
        network.push_back(std::vector<Perceptron>());
        if (i > 0)   //network[0] is the input layer,so it has no neurons
            for (size_t j = 0; j < layers[i]; j++)
                network[i].push_back(Perceptron(layers[i-1], bias));
    }
}


// Challenge: Finish the set_weights() and run() functions:


void MultiLayerPerceptron::set_weights(std::vector<std::vector<std::vector<double> > > w_init) {
    // Write all the weights into the neural network.
    // w_init is a vector of vectors of vectors of doubles. 
}

void MultiLayerPerceptron::print_weights() {
    std::cout << std::endl;
    for (size_t i = 1; i < network.size(); i++){
        for (size_t j = 0; j < layers[i]; j++) {
            std::cout << "Layer " << i+1 << " Neuron " << j << ": ";
            for (auto &it: network[i][j].weights)
                std::cout << it << "   ";
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

std::vector<double> MultiLayerPerceptron::run(std::vector<double> x) {
    // Run an input forward through the neural network.
    // x is a vector with the input values.
    return values.back();
}
