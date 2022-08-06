#include "yolov5.h"



using namespace cv;

/************************* model configuration *********************************/
#define DEVICE 0
// model config
const string model_path = "../../yolov5s.engine";
int inputw = 640;
int inputh = 640;
float nms_thresh = 0.4;
float conf_thresh = 0.5;


int main (int argc, char** argv) {
    // load model
    cudaSetDevice(DEVICE);
    printf("Initial memory:");
    printMemInfo();
    Yolov5 detector(model_path, inputw, inputh, nms_thresh, conf_thresh);
    cout << "create engine ";
    printMemInfo();

    Mat img, img0;

    // get input image (video)
    const string input_path = string(argv[1]);
    const string output_path = string(argv[2]);
    string f_ = input_path.substr(input_path.find_last_of(".") + 1);

    if (f_ == "jpg" || f_ == "png" || f_ == "jpeg") {
        img = imread(input_path);
        img0 = img.clone();
        if (img.empty()) cout << "read img fail" << endl;
        detector.doInfer(img0);    
        auto res = detector.result[0];
        for (int i = 0; i < res.size(); i++)
            rectangle(img, res[i].rect, Scalar(0,0,255), 2);

        imwrite(output_path, img);
    }

    else if (f_ == "mp4" || f_ == "avi") {
        VideoCapture cap(input_path);
        if (!cap.isOpened()) {
            cout << "video is empty" << endl;
            return 0;
        }

        int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
        int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
        int fps = cap.get(CAP_PROP_FPS);
        VideoWriter writer(output_path, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));


        int frame_id = 0, total_ms = 0;
        while (cap.read(img)) {
            frame_id++;
            if (frame_id % 50 == 0)
            {
                cout << "Processing frame " << frame_id << " (" << frame_id * 1000000 / total_ms << " fps)" << endl;
            }
            if (img.empty()) break;

            auto start = chrono::system_clock::now();

            img0 = img.clone();
            detector.doInfer(img0);
            
            auto end = chrono::system_clock::now();
            total_ms += chrono::duration_cast<chrono::microseconds>(end-start).count();

            auto res = detector.result[0];
            for (int i = 0; i < res.size(); i++)
            {
                rectangle(img, res[i].rect, Scalar(0,0,255), 2);
            }
            putText(img, format("frame: %d fps: %d num: %d", frame_id, frame_id * 1000000 / total_ms, res.size()), Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
            writer.write(img);

        }

        cap.release();
        cout << "FPS: " << frame_id * 1000000 / total_ms << endl;
    }
    return 0;
}