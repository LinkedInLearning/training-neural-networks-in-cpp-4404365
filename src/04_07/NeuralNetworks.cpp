// NeuralNetworks.cpp : This file contains the 'main' function. Program execution begins and ends there.//

#include <iostream>
#include "MLP.h"

int main() {
    srand(time(NULL));
    rand();


    cout << "\n\n--------Logic Gate Example----------------\n\n";
    Perceptron *p = new Perceptron(2);

    //{10,10,-15} #AND
    //{15,15,-10}  #OR
    //{-15,-15,10}  #NOR
    //{-10,-10,15} #NAND

    p->set_weights({15,15,-10});

    cout << "Gate: "<<endl;
    cout<<p->run({0,0})<<endl;
    cout<<p->run({0,1})<<endl;
    cout<<p->run({1,0})<<endl;
    cout<<p->run({1,1})<<endl;

    cout<<"\n\n--------Hardcoded XOR Example----------------\n\n";
    MultiLayerPerceptron mlp = MultiLayerPerceptron({2,2,1});  //mlp
    mlp.set_weights({{{-10,-10,15},{15,15,-10}}, {{10,10,-15}}});
    cout << "Hard-coded weights:\n";
    mlp.print_weights();

    cout<<"XOR:"<<endl;
    cout<<"0 0 = "<<mlp.run({0,0})[0]<<endl;
    cout<<"0 1 = "<<mlp.run({0,1})[0]<<endl;
    cout<<"1 0 = "<<mlp.run({1,0})[0]<<endl;
    cout<<"1 1 = "<<mlp.run({1,1})[0]<<endl;


    //test code - Trained XOR
    cout<<"\n\n--------Trained XOR Example----------------\n\n";
    mlp = MultiLayerPerceptron({2,2,1});
    cout<<"Training Neural Network as an XOR Gate...\n";
    double MSE;
    for (int i = 0; i < 3000; i++){
        MSE = 0.0;
        MSE += mlp.bp({0,0},{0});
        MSE += mlp.bp({0,1},{1});
        MSE += mlp.bp({1,0},{1});
        MSE += mlp.bp({1,1},{0});
        MSE = MSE / 4.0;
        if (i % 100 == 0)
            cout<<"MSE = "<<MSE<<endl;
    }

    cout<<"\n\nTrained weights (Compare to hard-coded weights):\n\n";
    mlp.print_weights();

    cout<<"XOR:"<<endl;
    cout<<"0 0 = "<<mlp.run({0,0})[0]<<endl;
    cout<<"0 1 = "<<mlp.run({0,1})[0]<<endl;
    cout<<"1 0 = "<<mlp.run({1,0})[0]<<endl;
    cout<<"1 1 = "<<mlp.run({1,1})[0]<<endl;

}
