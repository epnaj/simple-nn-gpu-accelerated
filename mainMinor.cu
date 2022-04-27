#include <iostream>
#include <ctime>
#include <random>
#include <vector>
#include <chrono>
#include <iomanip>
#include "matrix.h"
#include "struct.h"
#include "mnistCsvReader.h"

void fillWithInx(double *someData, int someDataSize){
    for(int i=0; i<someDataSize; ++i){
        someData[i] = i;
    }
}


int main(){
    srand(time(NULL));
    std::vector<double> k = {0, 1, 2, 3, 4, 5}, ksqr, target;
    
    net newNet({2,5,5,2}, sigmoidActivation); // highly depends on multilayer[WEIGHT NUMBER] and number of neurons (and reps)
    k = {0, 1};
    int n = 100000;
    auto start = std::chrono::high_resolution_clock::now();
    //newNet.printNet();
    for(int i=0; i<n; ++i){
        k[0] = rand()%2;
        k[1] = rand()%2;
        //if(k[0]&&k[1]||(!k[0]&&!k[1]))
        if( (k[0]==0&&k[1]==0) || (k[0]==1&&k[1]==1) )
            target = {1, 0};
        else
            target = {0, 1};
        //std::cout << k[0] << " " << k[1] << " " <<  target[0] << " " << target[1] << std::endl;
        newNet.train(k, target);
        //cudaDeviceSynchronize();
        //newNet.printError();
        
        //if((i+1)%(n/10)==0)
            //std::cout << (i+1)/(n/100) << "%" << std::endl;
    }
    newNet.printNet();
    auto stop = std::chrono::high_resolution_clock::now();
    double delta = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "Czas = " << delta << " milisekund  = "<< std::fixed << std::setprecision(2) << delta/1000 << " sekund" << std::endl;
    std::cout << "FOR 0 0 \n";
    k[0] = 0, k[1] = 0;
    newNet.forwardPropagate(k);
    newNet.printLastLayer();
    std::cout << "FOR 1 0 \n";
    k[0] = 1, k[1] = 0;
    newNet.forwardPropagate(k);
    newNet.printLastLayer();
    std::cout << "FOR 0 1\n";
    k[0] = 0, k[1] = 1;
    newNet.forwardPropagate(k);
    newNet.printLastLayer();
    std::cout << "FOR 1 1 \n";
    k[0] = 1, k[1] = 1;
    newNet.forwardPropagate(k);
    newNet.printLastLayer();

    // DZIALA :D
    // wymaga utworzenia biasow


    return 0;
}