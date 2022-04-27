#ifndef matrix_h
#define matrix_h
#include <iostream>
#include <vector>
#include <ctime>
#include <random>

class naiveMx{
    private:
    double *d_arr;
    int cls_cols, cls_rows;
    public:
    naiveMx(int rows, int cols, std::string flag); //random, default(any string)
    ~naiveMx();
    void printMx();
    void fill(double *someData, int someDataSize);
    void fillWithOwnVector(std::vector<double> &v, int rows, int cols);
    void fillWithOwnVectorTrustedSize(double *vec, int rows, int cols);
    void fillWithRandomNumbers_minus_1_1();
    void transpose();
    int getCols();
    int getRows();
    double *getArr();
    double **getArrAdress(int rows, int cols);    
};

__global__ void mutilplyMatrixKernel(double *a, double *b, double *c, int aRows, int aCols, int bRows, int bCols);

__global__ void transposeNaiveKernel(double *a, double *b, int originalRows, int originalCols);

__global__ void addMatrixesKernel(double *a, double *b, double *c, int n);

void multiplyMatrixes(naiveMx &a, naiveMx &b, naiveMx &c); // 1st matrix is multiplied by 2nd one and put in 3rd

void addMatrixes(naiveMx &a, naiveMx &b, naiveMx &c);

#endif