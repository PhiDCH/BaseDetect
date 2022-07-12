#pragma once


// #include <fstream>  
#include <memory>
// #include <vector>

// #include <opencv2/opencv.hpp>
#include "NvInfer.h"            // use tensorrt api 
// #include "cuda_runtime.h"       // use cuda api
#include "logging.h"            // define trtLogging

using namespace std;
using namespace nvinfer1;


void printMemInfo();

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            cerr << "Cuda failure: " << ret << endl;\
            abort();\
        }\
    } while (0)


static Logger gLogger;


class Detector {
    private:
    int inputC=3, inputH=1, inputW=1, outputSize=1;
    int inputIndex=0, outputIndex=1, maxBatchSize=1;
    int img_w=1280, img_h=720;
    float scale=1.0;
    unique_ptr<ICudaEngine> engine{nullptr};
    unique_ptr<IExecutionContext> context{nullptr};
    cudaStream_t stream;

    void *buffers[2];
    float *inputHost;
    
    // vector<vector<outputType>> res;

    public:
    // initialize model param and load model to GPU
    Detector(const string modelPath);
    // delete model and buffer
    ~Detector();
    
    // virtual void Preprocess(cv::Mat& img);
    // virtual void Postprocess(float *outputHost);
    // virtual void DoInfer(cv::Mat& img);
};


