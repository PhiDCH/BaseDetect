#include "base_detect.h"

#include <opencv2/opencv.hpp>


struct alignas(float) Detection {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;
};

struct Yolov5Output {
    cv::Rect_<float> rect;
    float prod;
    int label;
};


class Yolov5 : public Detector {
    void preprocess(cv::Mat& img);
    void postprocess();
    void get_rect(vector<Detection>& res);

    public:
    vector<vector<Yolov5Output>> result;
    float nms_thresh=0.4, conf_thresh=0.5;
    Yolov5 (const string modelPath);
    Yolov5 (const string modelPath, float nms_thresh, float conf_thresh);

    void doInfer(cv::Mat& img);
};