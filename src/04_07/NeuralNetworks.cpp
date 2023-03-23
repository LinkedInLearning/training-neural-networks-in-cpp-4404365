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


    // Test code - Trained XOR
    std::cout << "\n\n--------Trained XOR Example----------------\n\n";
    mlp = MultiLayerPerceptron({2,2,1});
    std::cout << "Training Neural Network as an XOR Gate..." << std::endl;
    double MSE;
    for (int i = 0; i < 3000; i++){
        MSE = 0.0;
        MSE += mlp.bp({0,0},{0});
        MSE += mlp.bp({0,1},{1});
        MSE += mlp.bp({1,0},{1});
        MSE += mlp.bp({1,1},{0});
        MSE = MSE / 4.0;
        if (i % 100 == 0)
            std::cout << "MSE = " << MSE << std::endl;
    }

    std::cout << "\n\nTrained weights (Compare to hardcoded weights):\n\n";
    mlp.print_weights();

    std::cout << "XOR:" << std::endl;
    std::cout << "0 0 = " << mlp.run({0,0})[0] << std::endl;
    std::cout << "0 1 = " << mlp.run({0,1})[0] << std::endl;
    std::cout << "1 0 = " << mlp.run({1,0})[0] << std::endl;
    std::cout << "1 1 = " << mlp.run({1,1})[0] << std::endl;
}
