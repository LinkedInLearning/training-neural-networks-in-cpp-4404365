// NeuralNetworks.cpp : This file contains the 'main' function. Program execution begins and ends there.//

#include "MLP.h"

int main(){
    srand(time(NULL));
    rand();

    std::cout << "\n\n--------Logic Gate Example----------------\n\n";
    Perceptron p(2);

    p.set_weights({10,10,-15}); //AND

    std::cout << "Gate: " << std::endl;
    std::cout << p.run({0,0}) << std::endl;
    std::cout << p.run({0,1}) << std::endl;
    std::cout << p.run({1,0}) << std::endl;
    std::cout << p.run({1,1}) << std::endl;
}
