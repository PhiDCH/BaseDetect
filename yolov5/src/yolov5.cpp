#include "yolov5.h"


using namespace cv;


// preprocess
static Mat static_resize(Mat& img, int inputW, int inputH) {
    int w, h, x, y;
    float r_w = inputW / (img.cols*1.0);
    float r_h = inputH / (img.rows*1.0);
    if (r_h > r_w) {
        w = inputW;
        h = r_w * img.rows;
        x = 0;
        y = (inputH - h) / 2;
    } else {
        w = r_h* img.cols;
        h = inputH;
        x = (inputW - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(inputH, inputW, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

static void blobFromImage(Mat& pre_img, float *inputHost, int inputW, int inputH){
    for (int i = 0; i < inputH * inputW; i++) {
        inputHost[i] = pre_img.at<Vec3b>(i)[2] / 255.0;
        inputHost[i + inputH * inputW] = pre_img.at<Vec3b>(i)[1] / 255.0;
        inputHost[i + 2 * inputH * inputW] = pre_img.at<Vec3b>(i)[0] / 255.0;
    }
}

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


/************************* define model function here *********************/

void Yolov5::preprocess(Mat& img) {
    img_w = img.cols;
    img_h = img.rows;

    img = static_resize(img, inputW, inputH);
    blobFromImage(img, inputHost, inputW, inputH);
}

void Yolov5::postprocess() {
    nms(result[0], &outputHost[0], conf_thresh, nms_thresh);
}

Yolov5::Yolov5 (const string modelPath) : Detector(modelPath) {
    result.resize(maxBatchSize);
}

Yolov5::Yolov5 (const string modelPath, float nms_thresh, float conf_thresh) : Detector(modelPath) {
    result.resize(maxBatchSize);
    nms_thresh = nms_thresh;
    conf_thresh = conf_thresh;
}

void Yolov5::doInfer(Mat& img) {
    preprocess(img);
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputHost, maxBatchSize*inputC*inputH*inputW*sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(outputHost, buffers[outputIndex], outputSize*sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    postprocess();
}


#define DEVICE 0

const string modelPath = "../../yolov5s.engine";

int main () {
    cudaSetDevice(DEVICE);

    printf("Initial memory:");
    printMemInfo();
    Yolov5 det1(modelPath);
    cout << "create engine ";
    printMemInfo();

    Mat img = imread("../../zidane.jpg");

    det1.doInfer(img);

    cout << det1.result[0].size() << endl;

    return 0;
}