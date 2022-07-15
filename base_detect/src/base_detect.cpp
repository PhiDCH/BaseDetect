#include <fstream> 

#include "base_detect.h"


// using namespace std;
// using namespace nvinfer1;
// using namespace cv;


#define ONE_GBYTE (1024*1024*1024)
#define ONE_MBYTE (1024*1024)
void printMemInfo()
{ 
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s\n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;

    // printf(" GPU memory usage: used = %.2f GB, free = %.2f GB, total = %.2f GB\n", used_db/ONE_GBYTE, free_db/ONE_GBYTE, total_db/ONE_GBYTE);
    printf(" GPU memory usage: used = %.2f MB, free = %.2f MB, total = %.2f MB\n", used_db/ONE_MBYTE, free_db/ONE_MBYTE, total_db/ONE_MBYTE);
}


/************************* define model function here *********************/
Detector::Detector (const string modelPath) {
    ////////// deserialize model
    char *trtModelStream{nullptr};
    size_t size{0};
    ifstream file;
    file.open(modelPath, ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    unique_ptr<IRuntime> runtime{createInferRuntime(gLogger)};
    assert(runtime != nullptr);
    engine.reset(runtime->deserializeCudaEngine(trtModelStream, size));
    assert(engine != nullptr);
    context.reset(engine->createExecutionContext());
    assert(context != nullptr);
    delete[] trtModelStream;
    CHECK(cudaStreamCreate(&stream));

    ///////////// get model configuration
    assert(engine->getNbBindings() == 2);
    // inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    // outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    // maxBatchSize = engine->getMaxBatchSize();
    Dims dim = engine->getBindingDimensions(inputIndex);
    maxBatchSize = dim.d[0];
    inputC = dim.d[1], inputH = dim.d[2], inputW = dim.d[3];
    // cout << "input shape " << dim.nbDims << " inputC" << inputC << " inputH" << inputH << " inputW" << inputW << endl;

    dim = engine->getBindingDimensions(outputIndex);
    for (int i=1; i<dim.nbDims; i++) outputSize *= dim.d[i];
    // cout << "output shape " << dim.nbDims << " " << dim.d[0] << "x" << dim.d[1] << "x" << dim.d[2] << endl;
    // cout << "outputSize " << outputSize << endl;
    // cout << "Max Batch Size " << maxBatchSize << endl;

    // outputHost = new float[maxBatchSize*outputSize];
    CHECK(cudaMalloc(&buffers[inputIndex], maxBatchSize*inputH*inputW*inputC*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], maxBatchSize*outputSize*sizeof(float)));
    inputHost = new float[maxBatchSize*inputH*inputW*inputC];
    outputHost = new float[maxBatchSize*outputSize];
    // res.resize(maxBatchSize);
}

Detector::~Detector () {
    //////////// free buffer
    cudaStreamDestroy(stream);
    CHECK(cudaFree(this->buffers[this->inputIndex]));
    CHECK(cudaFree(this->buffers[this->outputIndex]));
    delete[] inputHost;
    delete[] outputHost;
} 


// /************************* model configuration ****************8*****************/
// #define DEVICE 0  // GPU id
// const string modelPath = "/home/phidch/Downloads/vision-packages/BaseDetect/bytetrack_s.engine";

// int main () {
//     ///// set device
//     cudaSetDevice(DEVICE);

//     if (1) {
//         printf("Initial memory:");
//         printMemInfo();
//         Detector Det1(modelPath);
//         cout << "create engine ";
//         printMemInfo();
//     }

//     cout << "destroy all ";
//     printMemInfo();

//     return 0;
// }
