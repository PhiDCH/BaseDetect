#include "yolox.h"

using namespace cv;

/************************* model configuration ****************8*****************/
#define DEVICE 0  // GPU id
const string modelPath = "../../bytetrack_s.engine";


int main () {
    cudaSetDevice(DEVICE);

    printf("Initial memory:");
    printMemInfo();
    BASE_DETECT::Detector<OUTPUT_TYPE> detector(modelPath);
    cout << "create engine ";
    printMemInfo();

    // Mat img = imread("/home/cros/catkin_ws/src/1.jpg");
    // if (img.empty()) cout << "read img fail" << endl;
    // detector.DoInfer(img);    

    const string input_video_path = "../../palace.mp4";

    VideoCapture cap(input_video_path);
    if (!cap.isOpened()) {
        cout << "video is empty" << endl;
        return 0;
    }

    int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    // long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    Mat img;
    VideoWriter writer("../../demo.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));


    // int frame_id = 0, total_ms = 0;
    while (cap.read(img)) {
        // frame_id++;
        // if (frame_id % 20 == 0)
        // {
        //     cout << "Processing frame " << frame_id << " (" << frame_id * 1000000 / total_ms << " fps)" << endl;
        // }
        if (img.empty()) break;

        Mat img0 = img.clone();
        detector.DoInfer(img0);
        // vector<STrack> output_stracks = tracker.update(detector.getResult()[0]);

        auto output_det = detector.getResult()[0];
        for (int i = 0; i < output_det.size(); i++)
		{
            rectangle(img, output_det[i].rect, Scalar(0,0,255), 2);
		}
        // putText(img, format("frame: %d fps: %d num: %d", 1, 1, 1), 
        //         Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        writer.write(img);

    }

    cap.release();

    return 0;
}