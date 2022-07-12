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
    Yolox (const string modelPath);
    // ~Yolox ();
    void doInfer(cv::Mat& img);
    vector<vector<OUTPUT_TYPE>> result;
};




