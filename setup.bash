export BASE_PATH='/home/robotic/Downloads/BaseDetect'

cd $BASE_PATH
cd base_detect
mkdir build
cd build && cmake .. && make && sudo make install && cd ../..

cd $BASE_PATH
cd yolox
mkdir build
cd build && cmake .. && make && sudo make install && cd ../..

cd $BASE_PATH
cd yolov5
mkdir build
cd build && cmake .. && make && sudo make install && cd ../..

cd $BASE_PATH
cd yolov6
mkdir build
cd build && cmake .. && make && sudo make install && cd ../..

cd $BASE_PATH
cd bytetrack
mkdir build
cd build && cmake .. && make && sudo make install && cd ../..

cd $BASE_PATH
cd test
mkdir build
cd build && cmake .. && make && cd ../..