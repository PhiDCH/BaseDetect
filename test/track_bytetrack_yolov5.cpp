#include <chrono>

#include "yolov5.h"
#include "BYTETracker.h"


vector<Object> convert_yolov5Output_2_bytetrackInput(vector<Yolov5Output>& res) {
    vector<Object> tmp;
    tmp.resize(res.size());
    for (int i=0; i< res.size(); i++) {
        tmp[i].rect = res[i].rect;
        tmp[i].label = res[i].label;
        tmp[i].prob = res[i].prob;
    }
    return tmp;
}


/************************* model configuration ****************8*****************/
#define DEVICE 0  // GPU id
const string modelPath = "../../yolov5s.engine";
const string input_video_path = "../../palace.mp4";
const string output_video_path = "../../demo.mp4";

int main () {
    cudaSetDevice(DEVICE);


    printf("Initial memory:");
    printMemInfo();
    Yolov5 detector(modelPath, 0.7, 0.1);
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

    BYTETracker tracker(fps, 30);
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

        auto res = convert_yolov5Output_2_bytetrackInput(detector.result[0]);
        vector<STrack> output_stracks = tracker.update(res);

        auto end = chrono::system_clock::now();
        total_ms += chrono::duration_cast<chrono::microseconds>(end-start).count();

        for (int i = 0; i < output_stracks.size(); i++)
		{
            vector<float> tlwh = output_stracks[i].tlwh;
            Scalar s = tracker.get_color(output_stracks[i].track_id);
            putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
                        0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
            rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2, LINE_AA);
		}
        putText(img, format("frame: %d fps: %d num: %d", frame_id, frame_id * 1000000 / total_ms, output_stracks.size()), Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        writer.write(img);
    }

    cap.release();
    cout << "FPS: " << frame_id * 1000000 / total_ms << endl;
    return 0;
}
