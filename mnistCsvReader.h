#ifndef mnist_csv_hpp
#define mnist_csv_hpp

#include <fstream>
#include <vector>
using namespace std;


class mnist_img{
    public:
    vector <vector <double>> images;
    vector <double> labels;
    void load(string name,int number);
    void read_digit(int number);
    ~mnist_img();
};

#endif