#include "MLP.h"

double frand(){
	return (2.0*(double)rand() / RAND_MAX) - 1.0;
}


// Return a new Perceptron object with the specified number of inputs (+1 for the bias).
Perceptron::Perceptron(int inputs, double bias){
	this->bias = bias;
	weights.resize(inputs+1);
	generate(weights.begin(),weights.end(),frand);
}

// Run the perceptron. x is a vector with the input values.
double Perceptron::run(vector<double> x){
	x.push_back(bias);
	double sum = inner_product(x.begin(),x.end(),weights.begin(),(double)0.0);
	return sigmoid(sum);
}

// Set the weights. w_init is a vector with the weights.
void Perceptron::set_weights(vector<double> w_init){
	weights = w_init;
}

// Evaluate the sigmoid function for the floating point input x.
double Perceptron::sigmoid(double x){
	return 1.0/(1.0 + exp(-x));
}


// Return a new MultiLayerPerceptron object with the specified parameters.
MultiLayerPerceptron::MultiLayerPerceptron(vector<int> layers, double bias, double eta) {
    this->layers = layers;
    this->bias = bias;
    this->eta = eta;

    for (int i = 0; i < layers.size(); i++){
        values.push_back(vector<double>(layers[i],0.0));
        network.push_back(vector<Perceptron>());
        if (i > 0)   //network[0] is the input layer,so it has no neurons
            for (int j = 0; j < layers[i]; j++)
                network[i].push_back(Perceptron(layers[i-1], bias));
    }
}


// Set the weights. w_init is a vector of vectors of vectors with the weights for all but the input layer.
void MultiLayerPerceptron::set_weights(vector<vector<vector<double> > > w_init) {
    for (int i = 0; i< w_init.size(); i++)
        for (int j = 0; j < w_init[i].size(); j++)
            network[i+1][j].set_weights(w_init[i][j]);
}

void MultiLayerPerceptron::print_weights() {
    cout << endl;
    for (int i = 1; i < network.size(); i++){
        for (int j = 0; j < layers[i]; j++) {
            cout << "Layer " << i+1 << " Neuron " << j << ": ";
            for (auto &it: network[i][j].weights)
                cout << it <<"   ";
            cout << endl;
        }
    }
    cout << endl;
}

// Feed a sample x into the MultiLayer Perceptron.
vector<double> MultiLayerPerceptron::run(vector<double> x) {
    values[0] = x;
    for (int i = 1; i < network.size(); i++)
        for (int j = 0; j < layers[i]; j++)
            values[i][j] = network[i][j].run(values[i-1]);
    return values.back();
}
