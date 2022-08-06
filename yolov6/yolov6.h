#pragma once

#include "base_detect.h"

#include <opencv2/opencv.hpp>


struct Yolov6Output
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};


class Yolov6 : public Detector {
    void preprocess(cv::Mat& img);
    void postprocess();

    public:
    float nms_thresh, bbox_conf_thresh;
    Yolov6(const string modelPath, int inputw, int inputh, float nms_thresh_=0.7, float bbox_conf_thresh_=0.1);
    void doInfer(cv::Mat& img);
    vector<vector<Yolov6Output>> result;
};




