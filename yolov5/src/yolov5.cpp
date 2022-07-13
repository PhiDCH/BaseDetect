#include "base_detect.h"

#include "preprocess.h"

#include <opencv2/opencv.hpp>

using namespace cv;

struct alignas(float) Detection {
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
        size_t size_img = img.cols*img.rows*3;
        memcpy(img_host, img.data, size_img);
        CHECK(cudaMemcpyAsync(img_device, img_host, size_img, cudaMemcpyHostToDevice, stream));
        float *buffer_idx = (float*)buffers[inputIndex];
        preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, inputW, inputH, stream);
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
        CHECK(cudaStreamCreate(&stream));
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
        float outputHost[maxBatchSize*outputSize];
        CHECK(cudaMemcpyAsync(outputHost, buffers[outputIndex], maxBatchSize*outputSize*sizeof(float), cudaMemcpyDeviceToHost, stream));
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