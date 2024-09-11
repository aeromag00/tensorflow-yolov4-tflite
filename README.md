# ODYOU v0.1 (Object Detector for YOUTUBE)
Reference from tensorflow-yolov4-tflite
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

ODYOU is the Obeject Detection program throgh YOUTUBe Stream vidio.

YOLOv4, YOLOv4-tiny Implemented in Tensorflow 2.3.0. 
Convert YOLO v4, YOLOv3, YOLO tiny .weights to .pb, .tflite and trt format for tensorflow, tensorflow lite, tensorRT.

Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT


### 실행 조건
* Windows OS
* 인터넷 접속 필요 (ping to 8.8.8.8) 


### 최소 사양 (초당 24 frame 이상 출력 기준)
* GPU 탑재 그래픽카드 RTX 2060 급 이상 (그래픽 전용 메모리 8G 이상)
* CPU i7급 이상, 실행 디스크 SSD 급 이상 읽기 속도
#### OpenCL 연산 라이브러리
* CUDA 10.1 (cuDNN 7.6) 사용
#### 성능 예시
* GPU GTX 1650 4G, Ryzen 7>       FPS: 16
* CPU i3, Intel HD 4G>            FPS: 1
* CPU pentium 이하, Intel HD 2G>  FPS: 실행 불가


### Prerequisites for Developing
* CUDA 10.1
* cuDNN 7.6(7.5 불가)
* python 3.7.0
* Tensorflow 2.3.0
* tensorflow .pb weight
* COCO 2017 datasets

### Performance
<p align="center"><img src="data/performance.png" width="640"\></p>

### Demo

```bash
# Convert 모델 darknet .weights to tensorflow SavedModel(.pb)
## yolov3
python save_model.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3-416 --input_size 416 --model yolov3 

## yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 

## yolov4-tiny
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny

# Run demo tensorflow
## yolov3
python detect.py --weights ./checkpoints/yolov3-416 --size 416 --model yolov3 --image ./data/kite.jpg
python detectvideo.py --weights ./checkpoints/yolov3-416 --size 416 --model yolov3 --video ./data/road.mp4

## yolov4
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --image ./data/kite.jpg
python detectvideo.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/road.mp4
python detectvideo.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/4명-겹침-원거리.MOV
python detectvideo.py --video ./data/4명-겹침-원거리.MOV
python detectvideo.py --url_select
python detectvideo.py --url https://www.youtube.com/watch?v=evDN0Sa4T68  #세종시 멧돼지
python detectvideo.py --url https://www.youtube.com/watch?v=fANsFnkaX-U  #멧돼지

python detect.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --image ./data/kite.jpg --tiny

```
If you want to run yolov3 or yolov3-tiny change ``--model yolov3`` in command

#### Output

##### Yolov4 original weight
<p align="center"><img src="result.png" width="640"\></p>

##### Yolov4 tflite int8
<p align="center"><img src="result-int8.png" width="640"\></p>

### Convert to tflite

```bash
# Save tf model for tflite converting
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 --framework tflite

# yolov4
python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416.tflite

# yolov4 quantize float16
python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416-fp16.tflite --quantize_mode float16

# yolov4 quantize int8
python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416-int8.tflite --quantize_mode int8 --dataset ./coco_dataset/coco/val207.txt

# Run demo tflite model
python detect.py --weights ./checkpoints/yolov4-416.tflite --size 416 --model yolov4 --image ./data/kite.jpg --framework tflite
```
Yolov4 and Yolov4-tiny int8 quantization have some issues. I will try to fix that. You can try Yolov3 and Yolov3-tiny int8 quantization 


### Convert to TensorRT
```bash# yolov3
python save_model.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf --input_size 416 --model yolov3
python convert_trt.py --weights ./checkpoints/yolov3.tf --quantize_mode float16 --output ./checkpoints/yolov3-trt-fp16-416

# yolov3-tiny
python save_model.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --input_size 416 --tiny
python convert_trt.py --weights ./checkpoints/yolov3-tiny.tf --quantize_mode float16 --output ./checkpoints/yolov3-tiny-trt-fp16-416

# yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4.tf --input_size 416 --model yolov4
python convert_trt.py --weights ./checkpoints/yolov4.tf --quantize_mode float16 --output ./checkpoints/yolov4-trt-fp16-416
```

### Evaluate on COCO 2017 Dataset
```bash
# run script in /script/get_coco_dataset_2017.sh to download COCO 2017 Dataset
# preprocess coco dataset
cd data
mkdir data/dataset
cd ..
cd scripts
python coco_convert.py --input ./coco/annotations/instances_val2017.json --output val2017.pkl
python coco_annotation.py --coco_path ./coco 
cd ..

# evaluate yolov4 model
#python evaluate.py --weights ./data/yolov4.weights                               # 실행 오류
python evaluate.py --weights ./checkpoints/yolov4-416
#cd mAP/extra
cd mAP\\extra
python remove_space.py    #rename class
cd ..
python main.py --output results_yolov4_tf   #CPU 소모
```
#### mAP50 on COCO 2017 Dataset

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3      | 55.43   | 52.32   |         |
| YoloV4      | 61.96   | 57.33   |         |

### Benchmark
```bash
rem cd utils
python benchmarks.py --size 416 --model yolov4 --weights ./data/yolov4.weights    # 실행 오류
```
#### TensorRT performance
 
| YoloV4 416 images/s |   FP32   |   FP16   |   INT8   |
|---------------------|----------|----------|----------|
| Batch size 1        | 55       | 116      |          |
| Batch size 8        | 70       | 152      |          |

#### Tesla P100

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3 FPS  | 40.6    | 49.4    | 61.3    |
| YoloV4 FPS  | 33.4    | 41.7    | 50.0    |

#### Tesla K80

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3 FPS  | 10.8    | 12.9    | 17.6    |
| YoloV4 FPS  | 9.6     | 11.7    | 16.0    |

#### Tesla T4

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3 FPS  | 27.6    | 32.3    | 45.1    |
| YoloV4 FPS  | 24.0    | 30.3    | 40.1    |

#### Tesla P4

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3 FPS  | 20.2    | 24.2    | 31.2    |
| YoloV4 FPS  | 16.2    | 20.2    | 26.5    |

#### Macbook Pro 15 (2.3GHz i7)

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3 FPS  |         |         |         |
| YoloV4 FPS  |         |         |         |

### Training your own model
```bash
# Prepare your dataset
# If you want to train from scratch:
cd core
In config.py set FISRT_STAGE_EPOCHS=0             
                                    """default: 
                                        FISRT_STAGE_EPOCHS=0
                                        SECOND_STAGE_EPOCHS=30
                                    total Step: 74,820
                                    """
# Run script:
python train.py

# Transfer learning: 
python train.py --weights ./data/yolov4.weights   #default .weight 로딩
                                    """ImageSize:416 #MaxInUse: 3,050,779,136 #~1300 STEPs
                                       ImageSize:386 #MaxInUse: - (약 20시간 소요) #74,821 STEPs
                                    """

```
The training performance is not fully reproduced yet, so I recommended to use Alex's [Darknet](https://github.com/AlexeyAB/darknet) to train your own data, then convert the .weights to tensorflow or tflite.


### STUDY
* [ ] Training code(train.py>> training.py)

### TODO
* [ ] YOLOv4 tflite on ios
* [ ] ciou
* [ ] Mosaic data augmentation
* [x] YOLOv4 tflite on android
* [x] Convert YOLOv4 to Keras.H5

### HISTORY of v0.1
* [x] Convert YOLOv4 to TensorRT
* [x] Training code
* [x] Update scale xy
* [x] Mish activation
* [x] yolov4 tflite version
* [x] yolov4 in8 tflite version for mobile


### DISTRIBUTER
* hunglc007-tensorflow-yolov4-tflite: (https://github.com/hunglc007/tensorflow-yolov4-tflite)
  - YOLOv4, YOLOv4-tiny, YOLOv3, YOLOv3-tiny Implemented in Tensorflow 2.0, Android. Convert

### References
  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  * [darknet](https://github.com/AlexeyAB/darknet)
  
   My project is inspired by these previous fantastic YOLOv3 implementations:
  * [Yolov3 tensorflow](https://github.com/YunYang1994/tensorflow-yolov3)
  * [Yolov3 tf2](https://github.com/zzh8829/yolov3-tf2)

