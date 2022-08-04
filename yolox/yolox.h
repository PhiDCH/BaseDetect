#pragma once

#include "base_detect.h"

#include <opencv2/opencv.hpp>


struct YoloxOutput
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};


class Yolox : public Detector {
    void preprocess(cv::Mat& img);
    void postprocess();

    public:
    float nms_thresh, bbox_conf_thresh;
    Yolox(const string modelPath, int inputw, int inputh, float nms_thresh_=0.7, float bbox_conf_thresh_=0.1);
    void doInfer(cv::Mat& img);
    vector<vector<YoloxOutput>> result;
};




