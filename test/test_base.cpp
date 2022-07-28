#include "base_detect.h"

using namespace std;


/************************* model configuration ****************8*****************/
#define DEVICE 0  // GPU id
const string modelPath = "/home/phidch/Downloads/vision-packages/BaseDetect/yolov6s.engine";

int main () {
    ///// set device
    cudaSetDevice(DEVICE);

    if (1) {
        printf("Initial memory:");
        printMemInfo();
        Detector Det1(modelPath);
        cout << "create engine ";
        printMemInfo();
    }

    cout << "destroy all ";
    printMemInfo();

    return 0;
}