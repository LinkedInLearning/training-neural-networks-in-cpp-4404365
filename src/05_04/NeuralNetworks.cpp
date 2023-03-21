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

    //test code - Segment Display Recognition System
    int epochs = 1000;
    MultiLayerPerceptron *sdrnn;
    
    sdrnn = new MultiLayerPerceptron({7,7,1});

    // Dataset for the 7 to 1 network
    for (int i = 0; i < epochs; i++){
        MSE = 0.0;
        MSE += sdrnn->bp({1,1,1,1,1,1,0}, {0.05}); //0 pattern
        MSE += sdrnn->bp({0,1,1,0,0,0,0}, {0.15}); //1 pattern
        MSE += sdrnn->bp({1,1,0,1,1,0,1}, {0.25}); //2 pattern
        MSE += sdrnn->bp({1,1,1,1,0,0,1}, {0.35}); //3 pattern
        MSE += sdrnn->bp({0,1,1,0,0,1,1}, {0.45}); //4 pattern
        MSE += sdrnn->bp({1,0,1,1,0,1,1}, {0.55}); //5 pattern
        MSE += sdrnn->bp({1,0,1,1,1,1,1}, {0.65}); //6 pattern
        MSE += sdrnn->bp({1,1,1,0,0,0,0}, {0.75}); //7 pattern
        MSE += sdrnn->bp({1,1,1,1,1,1,1}, {0.85}); //8 pattern
        MSE += sdrnn->bp({1,1,1,1,0,1,1}, {0.95}); //9 pattern
    }
    MSE /= 10.0;
    cout << endl << "7 to 1  network MSE: " << MSE << endl;


    // Dataset for the 7 to 10 network
    delete(sdrnn);
    sdrnn = new MultiLayerPerceptron({7,7,10});
    
    for (int i = 0; i < epochs; i++){
        MSE = 0.0;
        MSE += sdrnn->bp({1,1,1,1,1,1,0}, {1,0,0,0,0,0,0,0,0,0}); //0 pattern
        MSE += sdrnn->bp({0,1,1,0,0,0,0}, {0,1,0,0,0,0,0,0,0,0}); //1 pattern
        MSE += sdrnn->bp({1,1,0,1,1,0,1}, {0,0,1,0,0,0,0,0,0,0}); //2 pattern
        MSE += sdrnn->bp({1,1,1,1,0,0,1}, {0,0,0,1,0,0,0,0,0,0}); //3 pattern
        MSE += sdrnn->bp({0,1,1,0,0,1,1}, {0,0,0,0,1,0,0,0,0,0}); //4 pattern
        MSE += sdrnn->bp({1,0,1,1,0,1,1}, {0,0,0,0,0,1,0,0,0,0}); //5 pattern
        MSE += sdrnn->bp({1,0,1,1,1,1,1}, {0,0,0,0,0,0,1,0,0,0}); //6 pattern
        MSE += sdrnn->bp({1,1,1,0,0,0,0}, {0,0,0,0,0,0,0,1,0,0}); //7 pattern
        MSE += sdrnn->bp({1,1,1,1,1,1,1}, {0,0,0,0,0,0,0,0,1,0}); //8 pattern
        MSE += sdrnn->bp({1,1,1,1,0,1,1}, {0,0,0,0,0,0,0,0,0,1}); //9 pattern
    }
    MSE /= 10.0;
    cout << "7 to 10 network MSE: " << MSE << endl;

    
    // Dataset for the 7 to 7 network
    delete(sdrnn);
    sdrnn = new MultiLayerPerceptron({7,7,7});

    for (int i = 0; i < epochs; i++){
        MSE = 0.0;
        MSE += sdrnn->bp({1,1,1,1,1,1,0}, {1,1,1,1,1,1,0}); //0 pattern
        MSE += sdrnn->bp({0,1,1,0,0,0,0}, {0,1,1,0,0,0,0}); //1 pattern
        MSE += sdrnn->bp({1,1,0,1,1,0,1}, {1,1,0,1,1,0,1}); //2 pattern
        MSE += sdrnn->bp({1,1,1,1,0,0,1}, {1,1,1,1,0,0,1}); //3 pattern
        MSE += sdrnn->bp({0,1,1,0,0,1,1}, {0,1,1,0,0,1,1}); //4 pattern
        MSE += sdrnn->bp({1,0,1,1,0,1,1}, {1,0,1,1,0,1,1}); //5 pattern
        MSE += sdrnn->bp({1,0,1,1,1,1,1}, {1,0,1,1,1,1,1}); //6 pattern
        MSE += sdrnn->bp({1,1,1,0,0,0,0}, {1,1,1,0,0,0,0}); //7 pattern
        MSE += sdrnn->bp({1,1,1,1,1,1,1}, {1,1,1,1,1,1,1}); //8 pattern
        MSE += sdrnn->bp({1,1,1,1,0,1,1}, {1,1,1,1,0,1,1}); //9 pattern
    }
    MSE /= 10.0;
    cout << "7 to 7  network MSE: " << MSE << endl << endl;

}
