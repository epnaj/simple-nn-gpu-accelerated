#include "struct.h"

/*  auto start = std::chrono::high_resolution_clock::now();
    //code
    auto stop = std::chrono::high_resolution_clock::now();
    double delta = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "Czas fill = " << delta << " milisekund  = "<< std::fixed << std::setprecision(2) << delta/1000 << " sekund" << std::endl;
    for time measure
*/

net::net(std::vector <int> nn_struct, std::string activationFunctionType) :
netActivationFunction(activationFunctionType)
{
    for(int i=0; i<nn_struct.size()-1; ++i){
        nn.push_back(new naiveMx(1, nn_struct.at(i), "default"));
        nn.push_back(new naiveMx(nn_struct.at(i), nn_struct.at(i+1), "random"));
        delta.push_back(new naiveMx(1, nn_struct.at(i), "default"));
        biases.push_back(new naiveMx(1, nn_struct.at(i), "random"));
    }
    nn.push_back(new naiveMx(1, nn_struct.at(nn_struct.size()-1), "default"));
    delta.push_back(new naiveMx(1, nn_struct.at(nn_struct.size()-1), "default"));
    biases.push_back(new naiveMx(1, nn_struct.at(nn_struct.size()-1), "random"));
    cudaMalloc(&d_target, nn_struct.at(nn_struct.size()-1)*sizeof(double));
    if(netActivationFunction == sigmoidActivation){
        activationFunction = &net::applySigmoid;
    }
    if(netActivationFunction == reluActivation){
        activationFunction = &net::applyReLU;
    }
    if(netActivationFunction == tanhActivation){
        activationFunction = &net::applyTanh;
    }
}

net::~net(){
    nn.~vector();
    delta.~vector();
    biases.~vector();
    cudaFree(d_target);
}

void net::printNet(){
    for(int i=0; i<nn.size(); ++i){
        nn[i]->printMx();
        std::cout << std::endl;
    }
}

void net::printLastLayer(){
    nn[nn.size()-1]->printMx();
}

void net::printDelta(){
    for(int i=0; i<delta.size(); ++i){
        delta[i]->printMx();
        std::cout << std::endl;
    }
}

void net::forwardPropagate(std::vector<double> &data){
    if(data.size()!=nn[0]->getCols()){
        // first layer is always 1 x n
        std::cout << "Data and layers sizes don't match, returning\n";
        return;
    }
    fillLayerAt(data, 0);
    for(int i=0; i<nn.size()-2; i+=2){
        multiplyMatrixes(*nn[i], *nn[i+1], *nn[i+2]);
        addMatrixes(*nn[i+2], *biases[(i+2)/2], *nn[i+2]);
        (this->*activationFunction)(i+2);
    }
}

void net::forwardPropagate(double *d_data, int sizeOfData){
    if(sizeOfData!=nn[0]->getCols()){
        // first layer is always 1 x n
        std::cout << "Data and layers sizes don't match, returning\n";
        return;
    }
    fillLayerAt(d_data, 0);
    for(int i=0; i<nn.size()-2; i+=2){
        multiplyMatrixes(*nn[i], *nn[i+1], *nn[i+2]);
        addMatrixes(*nn[i+2], *biases[(i+2)/2], *nn[i+2]);
        (this->*activationFunction)(i+2);
    }
}

void net::fillLayerAt(std::vector<double> &v, int layerIndex){
        nn[layerIndex]->fillWithOwnVector(v, nn[layerIndex]->getRows(), nn[layerIndex]->getCols());
        //guarantees same shape of any layer
        //here we do in trusted mode
}

void net::fillLayerAt(double *d_data, int layerIndex){
    if(layerIndex < nn.size())
        nn[layerIndex]->fillWithOwnVectorTrustedSize(d_data, nn[layerIndex]->getRows(), nn[layerIndex]->getCols());
}

void net::getError(std::vector <double> &target){
    cudaMemcpy(d_target, &target[0], target.size()*sizeof(double), cudaMemcpyHostToDevice);
    int n = target.size();
    int nOfThreads = 32;
    int nOfBlocks = (nOfThreads + n - 1)/nOfThreads;
    dim3 blocks(nOfBlocks, nOfBlocks);
    dim3 threads(nOfThreads, nOfThreads);
    getError_KERNEL <<<blocks, threads>>> (d_target, nn[nn.size()-1]->getArr(), delta[delta.size()-1]->getArr(), n);
}

void net::getError(double *d_targetMain, int targetSize){
    int nOfThreads = 32;
    int nOfBlocks = (nOfThreads + targetSize - 1)/nOfThreads;
    dim3 blocks(nOfBlocks, nOfBlocks);
    dim3 threads(nOfThreads, nOfThreads);
    getError_KERNEL <<<blocks, threads>>> (d_target, nn[nn.size()-1]->getArr(), delta[delta.size()-1]->getArr(), targetSize);
}

void net::printError(){
    std::cout << "PRINTING ERROR VALUE\n";
    double *helper = new double [nn[nn.size()-1]->getCols()];
    cudaMemcpy(helper, delta[delta.size()-1]->getArr(), nn[nn.size()-1]->getCols()*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i=0; i<nn[nn.size()-1]->getCols(); ++i){
        std::cout << helper[i] << " ";
    }
    std::cout << std::endl;
    delete [] helper;
}

std::vector<double> net::getLastLayerVals(){
    std::vector<double> toReturn;
    double *temporaryCopyingArr = new double [ nn[nn.size()-1]->getRows() * nn[nn.size()-1]->getCols() ];
    cudaMemcpy(temporaryCopyingArr, nn[nn.size()-1]->getArr(), nn[nn.size()-1]->getRows() * nn[nn.size()-1]->getCols()*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i=0; i<nn[nn.size()-1]->getRows() * nn[nn.size()-1]->getCols(); ++i){
        toReturn.push_back(temporaryCopyingArr[i]);
    }
    delete [] temporaryCopyingArr;
    return toReturn;
}

void net::applySigmoid(int layerIndex){
    int n = nn[layerIndex]->getRows()*nn[layerIndex]->getCols();
    int nOfThreads = 32;
    int nOfBlocks = (nOfThreads + n - 1)/nOfThreads;
    dim3 threads(nOfThreads, nOfThreads);
    dim3 blocks(nOfBlocks, nOfBlocks);
    sigmoid_KERNEL <<<nOfBlocks, nOfThreads>>> (nn[layerIndex]->getArr(), n);
}

void net::applyReLU(int layerIndex){
    int n = nn[layerIndex]->getRows()*nn[layerIndex]->getCols();
    int nOfThreads = 32;
    int nOfBlocks = (nOfThreads + n - 1)/nOfThreads;
    dim3 threads(nOfThreads, nOfThreads);
    dim3 blocks(nOfBlocks, nOfBlocks);
    relu_KERNEL <<<nOfBlocks, nOfThreads>>> (nn[layerIndex]->getArr(), n);
}

void net::applyTanh(int layerIndex){
    int n = nn[layerIndex]->getRows()*nn[layerIndex]->getCols();
    int nOfThreads = 32;
    int nOfBlocks = (nOfThreads + n - 1)/nOfThreads;
    dim3 threads(nOfThreads, nOfThreads);
    dim3 blocks(nOfBlocks, nOfBlocks);
    tanh_KERNEL <<<nOfBlocks, nOfThreads>>> (nn[layerIndex]->getArr(), n);
}

void net::backwardPropagate(){
    if(nn.size()<3) return;

    for(int i=delta.size()-1; i>0; --i){
        nn[2*i-1]->transpose();
        multiplyMatrixes( *delta[i], *nn[2*i-1], *delta[i-1] );
        nn[2*i-1]->transpose();
    }
    int nOfThreads = 32;   
    dim3 blocks(0, 0);
    dim3 threads(nOfThreads, nOfThreads);     
    if(netActivationFunction==sigmoidActivation){
        for(int i=1; i<delta.size(); ++i){
            blocks.x = (nOfThreads + nn[2*i-1]->getRows())/nOfThreads, blocks.y = (nOfThreads + nn[2*i-1]->getCols())/nOfThreads;
            justChainRule_KERNEL <<<blocks, threads>>>(nn[2*i-1]->getArr(), delta[i]->getArr(), nn[2*i]->getArr(), nn[2*(i-1)]->getArr(), learningRate, nn[2*i-1]->getRows(), nn[2*i-1]->getCols());
            blocks.x = 1;
            blocks.y = (nOfThreads + biases[i]->getCols() -1) / nOfThreads;
            updateBiases_KERNEL <<<blocks, threads>>>(biases[i]->getArr(), delta[i]->getArr(), nn[2*i]->getArr(), learningRate, biases[i]->getCols());
        }
    }
    if(netActivationFunction==reluActivation){
        for(int i=1; i<delta.size(); ++i){
            blocks.x = (nOfThreads + nn[2*i-1]->getRows())/nOfThreads, blocks.y = (nOfThreads + nn[2*i-1]->getCols())/nOfThreads;
            chainRuleReLU_KERNEL <<<blocks, threads>>> (nn[2*i-1]->getArr(), delta[i]->getArr(), nn[2*i]->getArr(), nn[2*(i-1)]->getArr(), learningRate, nn[2*i-1]->getRows(), nn[2*i-1]->getCols());
            blocks.x = 1;
            blocks.y = (nOfThreads + biases[i]->getCols() -1) / nOfThreads;
            updateBiasesReLU_KERNEL <<<blocks, threads>>> (biases[i]->getArr(), delta[i]->getArr(), nn[2*i]->getArr(), learningRate, biases[i]->getCols());
        }
    }
    if(netActivationFunction==tanhActivation){
        for(int i=1; i<delta.size(); ++i){
            blocks.x = (nOfThreads + nn[2*i-1]->getRows())/nOfThreads, blocks.y = (nOfThreads + nn[2*i-1]->getCols())/nOfThreads;
            chainRuleTanh_KERNEL <<<blocks, threads>>> (nn[2*i-1]->getArr(), delta[i]->getArr(), nn[2*i]->getArr(), nn[2*(i-1)]->getArr(), learningRate, nn[2*i-1]->getRows(), nn[2*i-1]->getCols());
            blocks.x = 1;
            blocks.y = (nOfThreads + biases[i]->getCols() -1) / nOfThreads;
            updateBiasesTanh_KERNEL <<<blocks, threads>>> (biases[i]->getArr(), delta[i]->getArr(), nn[2*i]->getArr(), learningRate, biases[i]->getCols());
        }
    }
}

void net::train(std::vector<double> &data, std::vector<double> &target){
    forwardPropagate(data);
    getError(target);
    backwardPropagate();
}

void net::train(double *d_data, int sizeOfData, double *d_targetMain, int targetSize){
    forwardPropagate(d_data, sizeOfData);
    getError(d_targetMain, targetSize);
    backwardPropagate();
}

void net::train(double *d_data, int sizeOfData, std::vector<double> &target){
    forwardPropagate(d_data, sizeOfData);
    getError(target);
    backwardPropagate();
}

__global__ void getError_KERNEL(double *a, double *b, double *c, int n){
    //a - output, b - target, c - diffrence
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n ){
        c[i] = 2*( a[i] - b[i] );
    }
}

__global__ void sigmoid_KERNEL(double *x, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n ){
        //x[i] = (1/(1+__expf(-x[i])));
        x[i] = (1/(1+exp(-x[i])));
    }
}

__global__ void sigmoidDerivative_KERNEL(double *x, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n ){
        x[i] *= (1-x[i]);
    }
}

__global__ void relu_KERNEL(double *x, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n ){
        x[i] = fmax(0.0, x[i]);
    }
}

__global__ void tanh_KERNEL(double *x, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n ){
        x[i] = (__expf(2*x[i])-1)/(__expf(2*x[i])+1);
    }
}

__global__ void justChainRule_KERNEL(double *x, double *delta,  double *value, double *beforeLayerValues, double learningRate, int rows, int cols){
    // delta size = value size, x size = n
    int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < rows && j < cols ){
        x[i*cols + j] +=  learningRate * delta[j] * beforeLayerValues[i] * (value[j]*(1-value[j]));
    }
}

__global__ void chainRuleReLU_KERNEL(double *x, double *delta,  double *value, double *beforeLayerValues, double learningRate, int rows, int cols){
    // delta size = value size, x size = n
    int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < rows && j < cols ){
        x[i*cols + j] +=  value[j] < 0 ? 0 : learningRate * delta[j] * beforeLayerValues[i];
    }
    // in relu div is 1
}

__global__ void chainRuleTanh_KERNEL(double *x, double *delta,  double *value, double *beforeLayerValues, double learningRate, int rows, int cols){
    // delta size = value size, x size = n
    int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < rows && j < cols ){
        x[i*cols + j] +=  learningRate * delta[j] * beforeLayerValues[i] * (1 - value[j]*value[j]);
    }
}

__global__ void updateBiases_KERNEL(double *biases, double *delta, double *values, double learningRate, int cols){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // 1 x n layer (same as values)
    if(j < cols){
        biases[j] +=  learningRate * delta[j] * (values[j]*(1-values[j]));
    }
}

__global__ void updateBiasesReLU_KERNEL(double *biases, double *delta, double *values, double learningRate, int cols){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // 1 x n layer (same as values)
    if(j < cols){
        biases[j] += values[j] < 0 ? 0 : learningRate * delta[j];
    }
}

__global__ void updateBiasesTanh_KERNEL(double *biases, double *delta, double *values, double learningRate, int cols){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j < cols){
        biases[j] +=  learningRate * delta[j] * (1 - values[j]*values[j]);
    }
}