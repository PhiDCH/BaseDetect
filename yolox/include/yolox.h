#pragma once

#include "base_detect.h"

#include <opencv2/opencv.hpp>


struct YoloxOutput
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
typedef YoloxOutput OUTPUT_TYPE;


class Yolox : public Detector {
    void preprocess(cv::Mat& img);
    void postprocess(float* outputHost);

    public:
    float nms_thresh=0.7, bbox_conf_thresh=0.1;
    Yolox(const string modelPath, float nms_thresh, float bbox_conf_thresh);
    Yolox(const string modelPath);
    void doInfer(cv::Mat& img);
    vector<vector<OUTPUT_TYPE>> result;
};




