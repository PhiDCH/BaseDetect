#include "base_detect.h"

using namespace std;


/************************* model configuration ****************8*****************/
#define DEVICE 0  // GPU id

int main (int argc, char** argv) {
    string model_path = string(argv[1]);

    cudaSetDevice(DEVICE);

    if (1) {
        printf("Initial memory:");
        printMemInfo();
        Detector Det1(model_path);
        cout << "create engine ";
        printMemInfo();
    }

    cout << "destroy all ";
    printMemInfo();

    return 0;
}