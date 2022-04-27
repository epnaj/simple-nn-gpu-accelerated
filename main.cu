#include <iostream>
#include <ctime>
#include <random>
#include <vector>
#include <chrono>
#include <iomanip>
#include "matrix.h"
#include "struct.h"
#include "mnistCsvReader.h"

int main(){
    int nOfTrainingData = 60000, nOfTestData = 10000, index;
    // IF nOfTestData IS SMALLER THAN 100 DELETE PROGRESS DISPLAY    
    std::vector<double> ll, target = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    net nnet( {784, 700, 700, 10}, sigmoidActivation);
    mnist_img m, m_two;
    m.load("mnistFiles/mnist_train.csv", nOfTrainingData);
    std::cout << "Data loaded\n";
    for(int loop=0; loop<3; ++loop){
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<nOfTrainingData; ++i){
        target[(int)m.labels[i]] = 1;
        nnet.train(m.images[i], target);
        target[(int)m.labels[i]] = 0;
        if((i+1)%(nOfTrainingData/10)==0)
            std::cout << (i+1)/(nOfTrainingData/100) << "%" << std::endl;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    double delta = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "Czas = " << delta << " milisekund  = " << std::fixed << std::setprecision(2) << delta/1000 << " sekund" << std::endl;
}
    std::cout << "TRAINING IS OVER\n";
    m_two.load("mnistFiles/mnist_test.csv", nOfTestData);
    std::cout << "Data loaded\n";
    double mx=0,good=0, mj;
    for(int i=0;i<nOfTestData;i++){
        nnet.forwardPropagate(m_two.images[i]);
        mx=0,mj=0;
        ll = nnet.getLastLayerVals();
        for(int j=0;j<ll.size();j++){
            if(mx<ll[j]){
                mx=ll[j];
                mj=j;
            }
        }
        if(mj==m_two.labels[i])
            good++;
    }
    std::cout << "Good : " << good << " Ratio = " << double(good/nOfTestData) << std::endl;
    for(int i=0; i<5; ++i){
        std::cout << "wprowadz liczbe miedzy 0 a " << nOfTestData-1 << std::endl;
        std::cin >> index;
        if(0<=index && index<nOfTestData){
            nnet.forwardPropagate(m_two.images[index]);
            m_two.read_digit(index);
            nnet.printLastLayer();
            std::cout << "etykieta: " << m_two.labels[index] << std::endl;
        }
    }
    return 0;
}