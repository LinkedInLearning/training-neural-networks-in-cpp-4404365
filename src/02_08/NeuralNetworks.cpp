// NeuralNetworks.cpp : This file contains the 'main' function. Program execution begins and ends there.//

#include <iostream>
#include "MLP.h"

int main() {
    srand(time(NULL));
    rand();


    cout << "\n\n--------Logic Gate Example----------------\n\n";
    Perceptron *p = new Perceptron(2);

    // Challenge: Write your OR weights
    p->set_weights({10,10,-15}); //AND

    cout << "Gate: "<<endl;
    cout<<p->run({0,0})<<endl;
    cout<<p->run({0,1})<<endl;
    cout<<p->run({1,0})<<endl;
    cout<<p->run({1,1})<<endl;

}
