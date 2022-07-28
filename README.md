# BaseDetect
This repo provide code for newly AI model serving written in Tensorrt APIs (C++).

[Yolov5](https://github.com/ultralytics/yolov5) [Yolov6](https://github.com/meituan/YOLOv6) [Yolov7](https://github.com/jinfagang/yolov7) [YoloX](https://github.com/Megvii-BaseDetection/YOLOX)

## Dependencies 
+ tensorrt
+ opencv
+ eigen

## Usage

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

Or simply run setup bash file 
```
bash setup.bash
```

Finally, test model
```
cd base_detect/build && cmake .. && make
./det_yolox
./track_bytetrack_yolox
```


## Download assets and checkpoints [here](https://drive.google.com/drive/folders/1XQ9Of7hJ32aYhHaY-k-g2B-mJVwb1xYb?usp=sharing) (tensorrt 8.4) 

