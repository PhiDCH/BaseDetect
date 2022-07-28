mkdir base_detect/build 
cd base_detect/build && cmake .. && make && sudo make install && cd ../..

mkdir yolox/build 
cd yolox/build && cmake .. && make && sudo make install && cd ../..

mkdir yolov5/build 
cd yolov5/build && cmake .. && make && sudo make install && cd ../..

mkdir bytetrack/build 
cd bytetrack/build && cmake .. && make && sudo make install && cd ../..

mkdir test/build 
cd test/build && cmake .. && make && cd ../..