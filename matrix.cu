#include "matrix.h"
#define MAX(a, b) ((a) > (b) ? (a) : (b))

naiveMx::naiveMx(int rows, int cols, std::string flag):cls_rows(rows), cls_cols(cols){
    cudaFree(d_arr);
    cudaMalloc(&d_arr, cols*rows*sizeof(double));
    if(flag=="random"){
        fillWithRandomNumbers_minus_1_1();
    }
}

naiveMx::~naiveMx(){
    cudaFree(d_arr);
}

void naiveMx::printMx(){
    if(!cls_cols || !cls_rows){
        std::cout << "MACIERZ JEST PUSTA\n";
        return;
    }
    double *a = new double [cls_cols*cls_rows];
    cudaMemcpy(a, d_arr, cls_cols*cls_rows*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i=0; i<cls_rows; ++i){
        for(int j=0; j<cls_cols; ++j){
            std::cout << *(a + i*cls_cols + j) << " ";
        }
        std::cout << std::endl;
    }
    delete [] a;
}

void naiveMx::fill(double *someData, int someDataSize){
    cudaMemcpy(d_arr, someData, someDataSize*sizeof(double), cudaMemcpyHostToDevice);
}

void naiveMx::fillWithOwnVector(std::vector<double> &v, int rows, int cols){
    if(v.size()<rows*cols) return;
    cudaMemcpy(d_arr, &v[0], v.size()*sizeof(double), cudaMemcpyHostToDevice);
    cls_rows = rows, cls_cols = cols;
    //if we delete temporaryCopyingArr here we also delete the vector, due to passig by reference
}

void naiveMx::fillWithOwnVectorTrustedSize(double *vec, int rows, int cols){
    cudaMemcpy(d_arr, vec, rows*cols*sizeof(double), cudaMemcpyDeviceToDevice);
}

void naiveMx::fillWithRandomNumbers_minus_1_1(){
    // unefficient should check for __host__
    double *temporaryCopyingArr = new double [cls_cols*cls_rows];
    for(int i=0; i<cls_cols*cls_rows; ++i){
        temporaryCopyingArr[i] = 2*(double)rand()/RAND_MAX - 1;
    }
    cudaMemcpy(d_arr, temporaryCopyingArr, cls_rows*cls_cols*sizeof(double), cudaMemcpyHostToDevice);
    delete [] temporaryCopyingArr;
}

void naiveMx::transpose(){
    double *temporaryCopyingArr;
    cudaMalloc(&temporaryCopyingArr, cls_rows*cls_cols*sizeof(double));
    int nOfThreads = 32;
    dim3 blocks((nOfThreads+cls_rows)/nOfThreads, (nOfThreads+cls_cols)/nOfThreads);
    dim3 threads(nOfThreads, nOfThreads);
    transposeNaiveKernel <<<blocks, threads>>>(d_arr, temporaryCopyingArr, cls_rows, cls_cols);
    cudaMemcpy(d_arr, temporaryCopyingArr, cls_rows*cls_cols*sizeof(double), cudaMemcpyDeviceToDevice);
    std::swap(cls_rows, cls_cols);
    cudaFree(temporaryCopyingArr);
}

int naiveMx::getCols(){ return cls_cols;}
int naiveMx::getRows(){ return cls_rows;}
double *naiveMx::getArr(){ return d_arr;   }
double **naiveMx::getArrAdress(int rows, int cols){
    cls_rows = rows, cls_cols = cols;
    return &d_arr;
}

__global__ void mutilplyMatrixKernel(double *a, double *b, double *c, int aRows, int aCols, int bRows, int bCols){
    // aCols = bRows !! alwawys
    // product has to be in shape of aRows x bCols
    int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < aRows && j < bCols ){
        double temp = 0;
        //#pragma unroll
        for(int k=0; k<aCols; k+=4){
            double4 doubleTemporary = reinterpret_cast<double4*>(&a[i*aCols + k])[0];
            temp += doubleTemporary.x * b[k*bCols +j];
            temp += doubleTemporary.y * b[(k+1)*bCols +j];
            temp += doubleTemporary.z * b[(k+2)*bCols +j];
            temp += doubleTemporary.w * b[(k+3)*bCols +j];
        }
        c[i*aCols + j] = temp;
    }
}

__global__ void transposeNaiveKernel(double *a, double *b, int originalRows, int originalCols){
    int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<originalRows && j<originalCols){
            b[j*originalRows + i] = a[originalCols*i + j];
    }
}

__global__ void addMatrixesKernel(double *a, double *b, double *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n ){
        c[i] = a[i] + b[i];
    }
}

void multiplyMatrixes(naiveMx &a, naiveMx &b, naiveMx &c){
    if(a.getCols() != b.getRows()){
        std::cout << "Zle parametry macierzy\n";
        return;
    }
    int n = a.getCols()*b.getRows();
    int nOfThreads = 32;
    dim3 blocks((nOfThreads+a.getRows())/nOfThreads, (nOfThreads+a.getCols())/nOfThreads);
    dim3 threads(nOfThreads, nOfThreads);
    mutilplyMatrixKernel <<<blocks, threads>>>(a.getArr(), b.getArr(), c.getArr(), a.getRows(), a.getCols(), b.getRows(), b.getCols());
}

void addMatrixes(naiveMx &a, naiveMx &b, naiveMx &c){
    if(a.getCols() != b.getCols() || b.getRows() != a.getRows() || c.getRows() != a.getRows() || c.getCols() != a.getCols()){
        std::cout << "BLAD DODAWANIA MACIERZY\n";
        return;
    }
    int n = a.getRows()*a.getCols();
    int nOfThreads = 32; //CHANGE
    int nOfBlocks = (nOfThreads + n - 1)/nOfThreads;
    dim3 blocks(nOfBlocks, nOfBlocks);
    dim3 threads(nOfThreads, nOfThreads);
    addMatrixesKernel <<<blocks, threads>>> (a.getArr(), b.getArr(), c.getArr(), n);
}