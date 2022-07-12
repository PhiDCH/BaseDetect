# BaseDetect
This repo provide code for newly AI model serving written in Tensorrt APIs (C++ and python). 


First, build base_detect:

```
cd base_detect/build && cmake .. && sudo make install && cd ../..
```
then build model, ex yolox:
```
cd yolox/build && cmake .. && sudo make install && cd ../..
```
and build tracking algorithm, ex bytetrack:
```
cd bytetrack/build && cmake .. && sudo make install && cd ../..
```

Finally, test model
```
cd base_detect/build && cmake .. && make
./det_yolox
./track_bytetrack_yolox
```