// NeuralNetworks.cpp : This file contains the 'main' function. Program execution begins and ends there.//

#include "MLP.h"

int main(){
    srand(time(NULL));
    rand();

    std::cout << "\n\n--------Logic Gate Example----------------\n\n";
    Perceptron p(2);

    // AND:  {10,10,-15}
    // OR:   {15,15,-10}
    // NOR:  {-15,-15,10}
    // NAND: {-10,-10,15}

    p.set_weights({15,15,-10});

    std::cout << "Gate: " << std::endl;
    std::cout << p.run({0,0}) << std::endl;
    std::cout << p.run({0,1}) << std::endl;
    std::cout << p.run({1,0}) << std::endl;
    std::cout << p.run({1,1}) << std::endl;

    std::cout << "\n\n--------Hardcoded XOR Example----------------\n\n";
    MultiLayerPerceptron mlp({2,2,1});
    mlp.set_weights({{{-10,-10,15},{15,15,-10}}, {{10,10,-15}}});
    std::cout << "Hardcoded weights:" << std::endl;
    mlp.print_weights();

    std::cout << "XOR:" << std::endl;
    std::cout << "0 0 = " << mlp.run({0,0})[0]<<std::endl;
    std::cout << "0 1 = " << mlp.run({0,1})[0]<<std::endl;
    std::cout << "1 0 = " << mlp.run({1,0})[0]<<std::endl;
    std::cout << "1 1 = " << mlp.run({1,1})[0]<<std::endl;
}
