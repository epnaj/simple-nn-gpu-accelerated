#ifndef struct_h
#define struct_h

#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include "matrix.h"
#define sigmoidActivation "sigmoidActivationFunction"
#define reluActivation "reluActivationFunction"
#define tanhActivation "tanhActivationFunction"

class net{
    private:
    std::vector <naiveMx*> nn;
    std::vector <naiveMx*> delta;
    std::vector <naiveMx*> biases;
    double *d_target;
    double learningRate = 0.001;
    void (net::*activationFunction)(int);
    const std::string netActivationFunction;
    public:
    net(std::vector <int> nn_struct, std::string activationFunctionType);
    ~net();
    void printNet();
    void printLastLayer();
    void printDelta();
    void forwardPropagate(std::vector<double> &data);
    void forwardPropagate(double *d_data, int sizeOfData);
    void fillLayerAt(std::vector<double> &v, int layerIndex);
    void fillLayerAt(double *d_data, int layerIndex);
    void getError(std::vector <double> &target);
    void getError(double *d_targetMain, int targetSize);
    void printError();
    std::vector<double> getLastLayerVals();
    void applySigmoid(int layerIndex);
    void applyReLU(int layerIndex);
    void applyTanh(int layerIndex);
    void backwardPropagate();
    void train(std::vector<double> &data, std::vector<double> &target);
    void train(double *d_data, int sizeOfData, double *d_targetMain, int targetSize);
    void train(double *d_data, int sizeOfData, std::vector<double> &target);
};

__global__ void getError_KERNEL(double *a, double *b, double *c, int n);

__global__ void sigmoid_KERNEL(double *x, int n);

__global__ void sigmoidDerivative_KERNEL(double *x, int n);

__global__ void relu_KERNEL(double *x, int n);

__global__ void tanh_KERNEL(double *x, int n);

__global__ void justChainRule_KERNEL(double *x, double *delta,  double *value, double *beforeLayerValues, double learningRate, int rows, int cols);

__global__ void chainRuleReLU_KERNEL(double *x, double *delta,  double *value, double *beforeLayerValues, double learningRate, int rows, int cols);

__global__ void chainRuleTanh_KERNEL(double *x, double *delta,  double *value, double *beforeLayerValues, double learningRate, int rows, int cols);

__global__ void updateBiases_KERNEL(double *biases, double *delta,  double *values, double learningRate, int cols);

__global__ void updateBiasesReLU_KERNEL(double *biases, double *delta, double *values, double learningRate, int cols);

__global__ void updateBiasesTanh_KERNEL(double *biases, double *delta, double *values, double learningRate, int cols);

#endif