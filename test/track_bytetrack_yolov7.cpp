#include <chrono>

#include "yolov6.h"
#include "BYTETracker.h"


vector<Object> convert_yolov6Output_2_bytetrackInput(vector<Yolov6Output>& res) {
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

int main (int argc, char** argv) {
    // model config
    const string model_path = "../../yolov7s.engine";
    // int inputw = 1088;
    // int inputh = 608;
    int inputw = 640;
    int inputh = 640;
    float nms_thresh = 0.7;
    float bbox_conf_thresh = 0.1;

    // load model
    cudaSetDevice(DEVICE);
    printf("Initial memory:");
    printMemInfo();
    Yolov6 detector(model_path, inputw, inputh, nms_thresh, bbox_conf_thresh);
    cout << "create engine ";
    printMemInfo();
  
    // get input image (video)
    const string input_path = string(argv[1]);
    const string output_path = string(argv[2]);

    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        cout << "video is empty" << endl;
        return 0;
    }

    int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);

    Mat img;
    VideoWriter writer(output_path, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));

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

        auto res = convert_yolov6Output_2_bytetrackInput(detector.result[0]);
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