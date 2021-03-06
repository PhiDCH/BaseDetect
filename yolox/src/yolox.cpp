#include "yolox.h"


using namespace cv;



// preprocess function
Mat static_resize(Mat& img, int inputW, int inputH, float scale) {
    int unpad_w = scale * img.cols;
    int unpad_h = scale * img.rows;
    Mat re(unpad_h, unpad_w, CV_8UC3);
    resize(img, re, re.size());
    Mat out(inputH, inputW, CV_8UC3, Scalar(114, 114, 114));
    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));
    return out;
}

void blobFromImage(Mat& img, float *inputHost){
    cvtColor(img, img, COLOR_BGR2RGB);

    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    vector<float> mean = {0.485, 0.456, 0.406};
    vector<float> std = {0.229, 0.224, 0.225};
    for (int c = 0; c < channels; c++) 
    {
        for (int  h = 0; h < img_h; h++) 
        {
            for (int w = 0; w < img_w; w++) 
            {
                inputHost[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<Vec3b>(h, w)[c]) / 255.0f - mean[c]) / std[c];
            }
        }
    }
}

// postprocess function
struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static void generate_grids_and_stride(int target_w, int target_h, vector<int>& strides, vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

static inline float intersection_area(YoloxOutput& a, YoloxOutput& b)
{
    Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(vector<YoloxOutput>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(vector<YoloxOutput>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(vector<YoloxOutput>& faceobjects, vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        YoloxOutput& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            YoloxOutput& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_yolox_proposals(vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, vector<YoloxOutput>& objects)
{
    const int num_class = 1;

    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (num_class + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                YoloxOutput obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}


/************************* define model function here *********************/
Yolox::Yolox(const string modelPath, float nms_thresh, float bbox_conf_thresh) : Detector(modelPath){
    result.resize(maxBatchSize);
    nms_thresh = nms_thresh;
    bbox_conf_thresh = bbox_conf_thresh;
}


Yolox::Yolox(const string modelPath) : Detector(modelPath){
    result.resize(maxBatchSize);
}


void Yolox::preprocess(Mat& img) {
    // resize
    img_w = img.cols;
    img_h = img.rows;
    scale = min(inputW / (img.cols*1.0), inputH / (img.rows*1.0));

    img = static_resize(img, inputW, inputH, scale);
    // normalize
    blobFromImage(img, inputHost);
}


void Yolox::postprocess(float *outputHost) {

    vector<YoloxOutput> proposals;
    vector<int> strides = {8, 16, 32};
    vector<GridAndStride> grid_strides;
    generate_grids_and_stride(inputW, inputH, strides, grid_strides);
    generate_yolox_proposals(grid_strides, outputHost,  bbox_conf_thresh, proposals);

    qsort_descent_inplace(proposals);

    vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_thresh);

    int count = picked.size();

    auto objects = result[0];
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}


void Yolox::doInfer(Mat& img) {
    preprocess(img);
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputHost, maxBatchSize*inputC*inputH*inputW*sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueueV2(buffers, stream, nullptr);
    float outputHost[maxBatchSize*outputSize];
    CHECK(cudaMemcpyAsync(outputHost, buffers[outputIndex], maxBatchSize*outputSize*sizeof(float), cudaMemcpyDeviceToHost, stream));
    postprocess(outputHost);
}


// /************************* model configuration ****************8*****************/
// #define DEVICE 0  // GPU id
// const string modelPath = "/home/phidch/Downloads/vision-packages/BaseDetect/bytetrack_s.engine";

// int main () {
//     ///// set device
//     cudaSetDevice(DEVICE);

//     printf("Initial memory:");
//     printMemInfo();
//     Yolox Det1(modelPath);
//     cout << "create engine ";
//     printMemInfo();



//     const string input_video_path = "../../palace.mp4";

//     VideoCapture cap(input_video_path);
//     if (!cap.isOpened()) {
//         cout << "video is empty" << endl;
//         return 0;
//     }

//     int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
// 	int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
//     int fps = cap.get(CAP_PROP_FPS);
//     // long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
//     Mat img;
//     VideoWriter writer("../../demo.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));


//     // int frame_id = 0, total_ms = 0;
//     while (cap.read(img)) {
//         // frame_id++;
//         // if (frame_id % 20 == 0)
//         // {
//         //     cout << "Processing frame " << frame_id << " (" << frame_id * 1000000 / total_ms << " fps)" << endl;
//         // }
//         if (img.empty()) break;

//         Mat img0 = img.clone();
//         Det1.doInfer(img0);
//         // vector<STrack> output_stracks = tracker.update(detector.getResult()[0]);

//         auto output_det = Det1.result[0];
//         for (int i = 0; i < output_det.size(); i++)
// 		{
//             rectangle(img, output_det[i].rect, Scalar(0,0,255), 2);
// 		}
//         // putText(img, format("frame: %d fps: %d num: %d", 1, 1, 1), 
//         //         Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
//         writer.write(img);

//     }

//     cap.release();

//     return 0;
// }