// NeuralNetworks.cpp : This file contains the 'main' function. Program execution begins and ends there.//

#include "MLP.h"
#include <sstream>

std::vector<double> read_vector(){
    std::vector<double> values = {0,1};
    std::string input;

    while(values.size() != 7){
        values = {};
        std::cout << "Input pattern \"a b c d e f g\": ";
        std::getline(std::cin, input);
        std::stringstream ss(input);
        double value;
        while(ss >> value){
            values.push_back(value);
            if (ss.peek() == ' ')
                ss.ignore();
        }
        if(values[0] < 0.0)
          break;
        if(values.size() != 7) 
            std::cout << "Error: Input must contain exactly 7 floating point values separated spaces." << std::endl;
    }
    return values;
}

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


    // Test code - Segment Display Recognition System

    std::cout << std::endl << "Segment Display Recognition System" << std::endl;
    std::cout << "How many epochs? ";
    size_t epochs;
    std::cin >> epochs;
    std::cin.ignore();
    MultiLayerPerceptron mlp1 = MultiLayerPerceptron({7,7,1});
    MultiLayerPerceptron mlp2 = MultiLayerPerceptron({7,7,10});
    MultiLayerPerceptron mlp3 = MultiLayerPerceptron({7,7,7});

    // Dataset for the 7 to 1 network
    std::cout << "Training 7 to 1 network..." << std::endl;
    for (int i = 0; i < epochs; i++){
        MSE = 0.0;
        MSE += mlp1.bp({1,1,1,1,1,1,0}, {0.05}); //0 pattern
        MSE += mlp1.bp({0,1,1,0,0,0,0}, {0.15}); //1 pattern
        MSE += mlp1.bp({1,1,0,1,1,0,1}, {0.25}); //2 pattern
        MSE += mlp1.bp({1,1,1,1,0,0,1}, {0.35}); //3 pattern
        MSE += mlp1.bp({0,1,1,0,0,1,1}, {0.45}); //4 pattern
        MSE += mlp1.bp({1,0,1,1,0,1,1}, {0.55}); //5 pattern
        MSE += mlp1.bp({1,0,1,1,1,1,1}, {0.65}); //6 pattern
        MSE += mlp1.bp({1,1,1,0,0,0,0}, {0.75}); //7 pattern
        MSE += mlp1.bp({1,1,1,1,1,1,1}, {0.85}); //8 pattern
        MSE += mlp1.bp({1,1,1,1,0,1,1}, {0.95}); //9 pattern
    }
    MSE /= 10.0;
    std::cout << std::endl << "7 to 1  network MSE: " << MSE << std::endl;


    // Dataset for the 7 to 10 network
    std::cout << "Training 7 to 10 network..." << std::endl;
    
    for (int i = 0; i < epochs; i++){
        MSE = 0.0;
        MSE += mlp2.bp({1,1,1,1,1,1,0}, {1,0,0,0,0,0,0,0,0,0}); //0 pattern
        MSE += mlp2.bp({0,1,1,0,0,0,0}, {0,1,0,0,0,0,0,0,0,0}); //1 pattern
        MSE += mlp2.bp({1,1,0,1,1,0,1}, {0,0,1,0,0,0,0,0,0,0}); //2 pattern
        MSE += mlp2.bp({1,1,1,1,0,0,1}, {0,0,0,1,0,0,0,0,0,0}); //3 pattern
        MSE += mlp2.bp({0,1,1,0,0,1,1}, {0,0,0,0,1,0,0,0,0,0}); //4 pattern
        MSE += mlp2.bp({1,0,1,1,0,1,1}, {0,0,0,0,0,1,0,0,0,0}); //5 pattern
        MSE += mlp2.bp({1,0,1,1,1,1,1}, {0,0,0,0,0,0,1,0,0,0}); //6 pattern
        MSE += mlp2.bp({1,1,1,0,0,0,0}, {0,0,0,0,0,0,0,1,0,0}); //7 pattern
        MSE += mlp2.bp({1,1,1,1,1,1,1}, {0,0,0,0,0,0,0,0,1,0}); //8 pattern
        MSE += mlp2.bp({1,1,1,1,0,1,1}, {0,0,0,0,0,0,0,0,0,1}); //9 pattern
    }
    MSE /= 10.0;
    std::cout << "7 to 10 network MSE: " << MSE << std::endl;

    
    // Dataset for the 7 to 7 network
    std::cout << "Training 7 to 7 network..." << std::endl;

    for (int i = 0; i < epochs; i++){
        MSE = 0.0;
        MSE += mlp3.bp({1,1,1,1,1,1,0}, {1,1,1,1,1,1,0}); //0 pattern
        MSE += mlp3.bp({0,1,1,0,0,0,0}, {0,1,1,0,0,0,0}); //1 pattern
        MSE += mlp3.bp({1,1,0,1,1,0,1}, {1,1,0,1,1,0,1}); //2 pattern
        MSE += mlp3.bp({1,1,1,1,0,0,1}, {1,1,1,1,0,0,1}); //3 pattern
        MSE += mlp3.bp({0,1,1,0,0,1,1}, {0,1,1,0,0,1,1}); //4 pattern
        MSE += mlp3.bp({1,0,1,1,0,1,1}, {1,0,1,1,0,1,1}); //5 pattern
        MSE += mlp3.bp({1,0,1,1,1,1,1}, {1,0,1,1,1,1,1}); //6 pattern
        MSE += mlp3.bp({1,1,1,0,0,0,0}, {1,1,1,0,0,0,0}); //7 pattern
        MSE += mlp3.bp({1,1,1,1,1,1,1}, {1,1,1,1,1,1,1}); //8 pattern
        MSE += mlp3.bp({1,1,1,1,0,1,1}, {1,1,1,1,0,1,1}); //9 pattern
    }
    MSE /= 10.0;
    std::cout << "7 to 7  network MSE: " << MSE << std::endl << std::endl;

    std::vector<double> pattern = {1.2};
    while(pattern[0] >= 0.0){
        pattern = read_vector();
        if(pattern[0] < 0.0)
            break;
        std::cout << "The number recognized by the 7 to 1 network is " << (int) (mlp1.run(pattern)[0]*10) << std::endl;

        auto temp = mlp2.run(pattern);
        auto max_iter = std::max_element(temp.begin(), temp.end());
        auto max_index = std::distance(temp.begin(), max_iter);
        std::cout << "The number recognized by the 7 to 10 network is " << max_index << std::endl;

        temp = mlp3.run(pattern);
        std::cout << "The pattern recognized by the 7 to 7 network is [";
        for (auto i : temp)
          std::cout << " " << i;
        std::cout << "]" << std::endl << std::endl;
    }
    std::cout << std::endl;
}



