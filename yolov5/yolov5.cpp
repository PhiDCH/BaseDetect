

#include "yolov5.h"
#include "yololayer.h"


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
    for (int i = 0; i < output[0] && i < 1000; i++) {
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
    vector<Detection> pre_res;
    nms(pre_res, outputHost, conf_thresh, nms_thresh);
    get_rect(pre_res);
}

void Yolov5::get_rect(vector<Detection>& pre_res) {
    auto& res = result[0];
    res.resize(pre_res.size());
    for (int i=0; i<pre_res.size(); i++) {
        res[i].prod = pre_res[i].conf;
        res[i].label = (int)pre_res[i].class_id;

        auto bbox = pre_res[i].bbox;
        int l, r, t, b;
        float r_w = inputW / (img_w * 1.0);
        float r_h = inputH / (img_h * 1.0);
        if (r_h > r_w) {
            l = bbox[0] - bbox[2] / 2.f;
            r = bbox[0] + bbox[2] / 2.f;
            t = bbox[1] - bbox[3] / 2.f - (inputH - r_w * img_h) / 2;
            b = bbox[1] + bbox[3] / 2.f - (inputH - r_w * img_h) / 2;
            l = l / r_w;
            r = r / r_w;
            t = t / r_w;
            b = b / r_w;
        } else {
            l = bbox[0] - bbox[2] / 2.f - (inputW - r_h * img_w) / 2;
            r = bbox[0] + bbox[2] / 2.f - (inputW - r_h * img_w) / 2;
            t = bbox[1] - bbox[3] / 2.f;
            b = bbox[1] + bbox[3] / 2.f;
            l = l / r_h;
            r = r / r_h;
            t = t / r_h;
            b = b / r_h;
        }
        res[i].rect = Rect_<float>(l, t, r - l, b - t);
    }
}

Yolov5::Yolov5 (const string modelPath, int inputw, int inputh, float nms_thresh_, float conf_thresh_) : Detector(modelPath) {
    result.resize(maxBatchSize);
    nms_thresh = nms_thresh_;
    conf_thresh = conf_thresh_;
    inputW = inputw;
    inputH = inputh;
}

void Yolov5::doInfer(Mat& img) {
    preprocess(img);
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputHost, maxBatchSize*inputSize*sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(outputHost, buffers[outputIndex], maxBatchSize*outputSize*sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    postprocess();
}




#define DEVICE 0

const string modelPath = "../../yolov5s.engine";
int inputw = 640;
int inputh = 640;
float nms_thresh = 0.4;
float conf_thresh = 0.5;

int main () {
    cudaSetDevice(DEVICE);


    printf("Initial memory:");
    printMemInfo();
    Yolov5 det(modelPath, inputw, inputh, nms_thresh, conf_thresh);
    cout << "create engine ";
    printMemInfo();

    Mat img = imread("../../zidane.jpg");
    Mat img0= img.clone();

    det.doInfer(img0);

    auto res = det.result[0];
    cout << res.size() << endl;
    for (int i=0; i<res.size(); i++) {
        rectangle(img, res[i].rect, Scalar(0,0,255), 2);
    }
    imwrite("../../test.jpg", img);


    return 0;
}