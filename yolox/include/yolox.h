#pragma once

#include <iostream> 
#include <opencv2/opencv.hpp>
#include <vector>

#include "base_detect.h"

using namespace cv;
using namespace std;

// output type
struct Object
{
    Rect_<float> rect;
    int label;
    float prob;
};
typedef Object OUTPUT_TYPE;

// preprocess function
Mat static_resize(Mat& img, int inputW, int inputH, float scale);

void blobFromImage(Mat& img, float *inputHost);

// postprocess function
struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static void generate_grids_and_stride(int target_w, int target_h, vector<int>& strides, vector<GridAndStride>& grid_strides);

static inline float intersection_area(Object& a, Object& b);

static void qsort_descent_inplace(vector<Object>& faceobjects, int left, int right);

static void qsort_descent_inplace(vector<Object>& objects);

static void nms_sorted_bboxes(vector<Object>& faceobjects, vector<int>& picked, float nms_threshold);

static void generate_yolox_proposals(vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, vector<Object>& objects);

void decode_outputs(float* prob, vector<Object>& objects, float scale, int img_w, int img_h, int inputW, int inputH);



