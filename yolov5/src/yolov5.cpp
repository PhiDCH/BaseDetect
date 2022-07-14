#include "base_detect.h"

#include "preprocess.h"

#include <opencv2/opencv.hpp>

using namespace cv;

#include <fstream>

using namespace std;
using namespace nvinfer1;
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

    ///////////// get model configuration
    assert(engine->getNbBindings() == 2);
    // inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    // outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    // maxBatchSize = engine->getMaxBatchSize();

    // cout << engine->getBindingName(0) << " " << engine->getBindingName(1) << endl;

    Dims dim = engine->getBindingDimensions(inputIndex);
    maxBatchSize = dim.d[0];
    inputC = dim.d[1], inputH = dim.d[2], inputW = dim.d[3];
    // cout << "input shape " << dim.nbDims << " inputC" << inputC << " inputH" << inputH << " inputW" << inputW << endl;

    dim = engine->getBindingDimensions(outputIndex);
    for (int i=1; i<dim.nbDims; i++) outputSize *= dim.d[i];
    // cout << "output shape " << dim.nbDims << " " << dim.d[0] << "x" << dim.d[1] << "x" << dim.d[2] << endl;
    // cout << "outputSize " << outputSize << endl;
    // cout << "Max Batch Size " << maxBatchSize << endl;
    inputHost = new float[maxBatchSize*inputH*inputW*inputC];
    
    CHECK(cudaMalloc(&buffers[inputIndex], maxBatchSize*inputH*inputW*inputC*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize*sizeof(float)));

    CHECK(cudaStreamCreate(&stream));
}

Detector::~Detector () {
    //////////// free buffer
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    delete inputHost;
    // delete outputHost;
} 



struct Detection {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;
};
typedef Detection Yolov5Output;

// preprocess
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000*3000

// postprocess
float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Detection& a, const Detection& b) {
    return a.conf > b.conf;
}

void nms(vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Detection) / sizeof(float);
    map<float, vector<Detection>> m;
    for (int i = 0; i < output[0]; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}


class Yolov5 : public Detector {
    void preprocess(Mat& img) {
        img_w = img.cols;
        img_h = img.rows;
        size_t size_img = 3*img_h*img_w;
        memcpy(img_host, img.data, size_img);
        CHECK(cudaMemcpyAsync(img_device, img_host, size_img, cudaMemcpyHostToDevice, stream));
        cout << img_device << endl;
        preprocess_kernel_img(img_device, img_w, img_h, (float*)buffers[inputIndex], inputW, inputH, stream);
    };

    void postprocess(float* outputHost) {
        auto& res = result[0];
        nms(res, outputHost, conf_thresh, nms_thresh);
    };

    public: 
    float nms_thresh=0.4, conf_thresh=0.5;
    uint8_t *img_host = nullptr;
    uint8_t *img_device = nullptr;
    
    vector<vector<Yolov5Output>> result;

    Yolov5 (const string modelPath) : Detector(modelPath) {
        CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH*3));
        CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH*3));
        result.resize(maxBatchSize);
        // CHECK(cudaStreamCreate(&stream));
    };

    Yolov5 (const string modelPath, float nms_thresh, float conf_thresh) : Detector(modelPath) {
        CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH*3));
        CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH*3));
        result.resize(maxBatchSize);
        CHECK(cudaStreamCreate(&stream));
        nms_thresh = nms_thresh;
        conf_thresh = conf_thresh;
    };

    ~Yolov5 () {
        CHECK(cudaFree(img_device));
        CHECK(cudaFreeHost(img_host));
    };

    void doInfer(Mat& img) {
        preprocess(img);
        context->enqueueV2(buffers, stream, nullptr);

        // cout << buffers[0] << " " << buffers[1] << endl;
        
        float outputHost[outputSize];
        CHECK(cudaMemcpyAsync(outputHost, buffers[outputIndex], outputSize*sizeof(float), cudaMemcpyDeviceToHost, stream));
        // cudaStreamSynchronize(stream);
        // postprocess(outputHost);
    };
};

#define DEVICE 0

const string modelPath = "../../yolov5s.engine";

int main () {
    cudaSetDevice(DEVICE);

    Yolov5 det1(modelPath);

    Mat img = imread("../../zidane.jpg");

    det1.doInfer(img);

    // cout << det1.result[0].size() << endl;

    return 0;
}