
#include <chrono>
#include <thread>

#include "base_detect.h"


using namespace std;
using namespace nvinfer1;
using namespace cv;

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.7
#define BBOX_CONF_THRESH 0.1


/************************* model configuration ****************8*****************/

const string modelPath = "/home/phidch/Downloads/vision-packages/bytetrack_s.engine";
// static const int INPUT_W = 640, INPUT_H = 480, INPUT_C = 3, OUTPUT_SIZE = 16;
// const char* INPUT_BLOB_NAME = "images", *OUTPUT_BLOB_NAME = "output";


/************************* define model function here *********************/
template <typename outputType>
void BASE_DETECT::Detector<outputType>::Preprocess(Mat& img) {
}

template <typename outputType>
void BASE_DETECT::Detector<outputType>::Postprocess(float *outputHost) {
}

template <typename outputType>
void BASE_DETECT::Detector<outputType>::DoInfer (Mat& img) {
    Preprocess(img);
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputHost, inputC*inputH*inputW*sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(maxBatchSize, buffers, stream, nullptr);
    float *outputHost;
    CHECK(cudaMemcpyAsync(outputHost, buffers[outputIndex], outputSize*sizeof(float), cudaMemcpyDeviceToHost, stream));
    Postprocess(outputHost);
}


struct outputType {
    int i;
};

int main () {
    /////// set device
    cudaSetDevice(DEVICE);
    // printf("Initial memory:");
    // printMemInfo();

    if (1) {
        printf("Initial memory:");
        printMemInfo();
        BASE_DETECT::Detector<outputType> Det1(modelPath);
        cout << "create engine ";
        printMemInfo();
    }

    cout << "destroy all ";
    printMemInfo();

    return 0;
}
