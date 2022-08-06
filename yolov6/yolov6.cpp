#include "yolov6.h"


using namespace cv;



// preprocess function
static Mat static_resize(Mat& img, int inputW, int inputH, float scale) {
    int unpad_w = scale * img.cols;
    int unpad_h = scale * img.rows;
    Mat re(unpad_h, unpad_w, CV_8UC3);
    resize(img, re, re.size());
    Mat out(inputH, inputW, CV_8UC3, Scalar(114, 114, 114));
    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));
    return out;
}

static void blobFromImage(Mat& img, float *inputHost){
    cvtColor(img, img, COLOR_BGR2RGB);

    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    // vector<float> mean = {0.485, 0.456, 0.406};
    // vector<float> std = {0.229, 0.224, 0.225};
    for (int c = 0; c < channels; c++) 
    {
        for (int  h = 0; h < img_h; h++) 
        {
            for (int w = 0; w < img_w; w++) 
            {
                inputHost[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
}

// postprocess function
static inline float intersection_area(Yolov6Output& a, Yolov6Output& b)
{
    Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(vector<Yolov6Output>& faceobjects, int left, int right)
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

static void qsort_descent_inplace(vector<Yolov6Output>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(vector<Yolov6Output>& faceobjects, vector<int>& picked, float nms_threshold)
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
        Yolov6Output& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            Yolov6Output& b = faceobjects[picked[j]];

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

static void generate_yolo_proposals(float* feat_blob, int output_size, float prob_threshold, std::vector<Yolov6Output>& objects)
{
    const int num_class = 80;
    auto dets = output_size / (num_class + 5);
    for (int boxs_idx = 0; boxs_idx < dets; boxs_idx++)
    {
        const int basic_pos = boxs_idx *(num_class + 5);
        float x_center = feat_blob[basic_pos+0];
        float y_center = feat_blob[basic_pos+1];
        float w = feat_blob[basic_pos+2];
        float h = feat_blob[basic_pos+3];
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;
        float box_objectness = feat_blob[basic_pos+4];
        // std::cout<<*feat_blob<<std::endl;
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Yolov6Output obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop
    }

}


/************************* define model function here *********************/
Yolov6::Yolov6(const string modelPath, int inputw, int inputh, float nms_thresh_, float bbox_conf_thresh_) : Detector(modelPath){
    result.resize(maxBatchSize);
    nms_thresh = nms_thresh_;
    bbox_conf_thresh = bbox_conf_thresh_;
    inputW = inputw;
    inputH = inputh;
}


void Yolov6::preprocess(Mat& img) {
    // resize
    img_w = img.cols;
    img_h = img.rows;
    scale = min(inputW / (img.cols*1.0), inputH / (img.rows*1.0));

    img = static_resize(img, inputW, inputH, scale);
    // normalize
    blobFromImage(img, inputHost);
}


void Yolov6::postprocess() {

    vector<Yolov6Output> proposals;
    generate_yolo_proposals(outputHost, outputSize, bbox_conf_thresh, proposals);

    qsort_descent_inplace(proposals);

    vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_thresh);

    int count = picked.size();

    auto& objects = result[0];
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


void Yolov6::doInfer(Mat& img) {
    preprocess(img);
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputHost, maxBatchSize*inputSize*sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(outputHost, buffers[outputIndex], maxBatchSize*outputSize*sizeof(float), cudaMemcpyDeviceToHost, stream));
    postprocess();
}




// /************************* model configuration ****************8*****************/
// #define DEVICE 0  // GPU id


// int main (int agrc, char** argv) {
//     const string model_path = "../../yolov6s.engine";
//     int inputw = 640;
//     int inputh = 640;
//     float nms_thresh = 0.7;
//     float conf_thresh = 0.1;
//     string input_path = "../../zidane.jpg";
//     string output_path = "../../test.jpg";

//     ///// set device
//     cudaSetDevice(DEVICE);

//     printf("Initial memory:");
//     printMemInfo();
//     Yolov6 Det1(model_path, inputw, inputh, nms_thresh, conf_thresh);
//     cout << "create engine ";
//     printMemInfo();

//     Mat img, img0;

//     img = imread(input_path);
//     img0 = img.clone();
//     Det1.doInfer(img0);
//     auto res = Det1.result[0];
//     for (int i = 0; i < res.size(); i++)
//         rectangle(img, res[i].rect, Scalar(0,0,255), 2);

//     imwrite(output_path, img);


//     // const string input_video_path = "../../palace.mp4";

//     // VideoCapture cap(input_video_path);
//     // if (!cap.isOpened()) {
//     //     cout << "video is empty" << endl;
//     //     return 0;
//     // }

//     // int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
// 	// int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
//     // int fps = cap.get(CAP_PROP_FPS);

//     // VideoWriter writer("../../demo.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));

//     // int frame_id = 0, total_ms = 0;
//     // while (cap.read(img)) {
//     //     frame_id++;
//     //     if (frame_id % 50 == 0)
//     //     {
//     //         cout << "Processing frame " << frame_id << " (" << frame_id * 1000000 / total_ms << " fps)" << endl;
//     //     }
//     //     if (img.empty()) break;

//     //     auto start = chrono::system_clock::now();

//     //     img0 = img.clone();
//     //     Det1.doInfer(img0);

//     //     auto end = chrono::system_clock::now();
//     //     total_ms += chrono::duration_cast<chrono::microseconds>(end-start).count();

//     //     auto output_det = Det1.result[0];
//     //     for (int i = 0; i < output_det.size(); i++)
// 	// 	{
//     //         rectangle(img, output_det[i].rect, Scalar(0,0,255), 2);
// 	// 	}
//     //     // putText(img, format("frame: %d fps: %d num: %d", 1, 1, 1), 
//     //     //         Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
//     //     writer.write(img);
//     // }

//     // cap.release();

//     return 0;
// }