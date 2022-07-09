#include "yolox.h"
#include "BYTETracker.h"


/************************* model configuration ****************8*****************/
#define DEVICE 0  // GPU id
const string modelPath = "../../bytetrack_s.engine";

int main () {
    cudaSetDevice(DEVICE);

    if (1) {
        printf("Initial memory:");
        printMemInfo();
        BASE_DETECT::Detector<OUTPUT_TYPE> Det1(modelPath);
        cout << "create engine ";
        printMemInfo();
    }

    cout << "destroy all ";
    printMemInfo();

    return 0;
}