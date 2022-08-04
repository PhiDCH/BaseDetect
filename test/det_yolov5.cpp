#include "yolov5.h"



using namespace cv;

/************************* model configuration ****************8*****************/
#define DEVICE 0  // GPU id
const string modelPath = "../../yolov5s.engine";
const string input_video_path = "../../palace.mp4";
const string output_video_path = "../../demo.mp4";

int main () {
    cudaSetDevice(DEVICE);


    printf("Initial memory:");
    printMemInfo();
    Yolov5 detector(modelPath);
    cout << "create engine ";
    printMemInfo();


    // Mat img = imread("/home/cros/catkin_ws/src/1.jpg");
    // if (img.empty()) cout << "read img fail" << endl;
    // detector.DoInfer(img);    


    VideoCapture cap(input_video_path);
    if (!cap.isOpened()) {
        cout << "video is empty" << endl;
        return 0;
    }

    int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    Mat img;
    VideoWriter writer(output_video_path, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));


    int frame_id = 0, total_ms = 0;
    while (cap.read(img)) {
        frame_id++;
        if (frame_id % 50 == 0)
        {
            cout << "Processing frame " << frame_id << " (" << frame_id * 1000000 / total_ms << " fps)" << endl;
        }
        if (img.empty()) break;

        auto start = chrono::system_clock::now();

        Mat img0 = img.clone();
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
    return 0;
}
