# DeepSort_with_yolov5
 * This Project is built on top of https://github.com/abhyantrika/nanonets_object_tracking and add yolov5 for detection of vehicals
 * By default it tracks Cars Trucks and Bus but can be expanded easily to cycles bikes also 
 * Siamese Network is trained on Nvidia AI City Challege Data and VeRI Wild Dataset

## Demo Video 
<br>
<div align="center">
Demo [<img src=".github/demo.png" width="50%">](https://drive.google.com/file/d/1aVlaJogjbz8Q8KUvc3_b4ybravZwiPGT/view?usp=sharing)
</div>

## Steps to run tracker on custom Video
 * Download Yolov5 weights from this link https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x6.pt
 * ```pip install -r requirements.txt ```
 * ```
    python custom_deepsort.py --yolo_weights PATH_TO_DOWNLOADED_WEIGHTS \
                            --source PATH_2_VIDEO_FILE \
                            --device GPU_ID_2_USE 
    ```
## Training steps 
 * Download dataset and save it in ``` object_tracking/datasets/train ``` and ``` object_tracking/datasets/test ``` (make sure format of data is correct i.e train/car_id/**images)
 * ```cd object_tracking``` and then  ```python siamese_train.py```
 * Default config is batch_size = 256, epoch = 40 and save_checkpoint after 10 epochs which can be altered as required 
 * loss will be saved at ```ckpts/loss.png``` after training is finished

## Get Test Scores 
 * Download dataset and save it in ``` object_tracking/datasets/train ``` and ``` object_tracking/datasets/test ``` (make sure format of data is correct i.e train/car_id/**images)
 * ```cd object_tracking``` and then  ```python siamese_test.py```
## Refrences 
 * Siamese Net https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
 * DeepSort Paper https://arxiv.org/abs/1703.07402
 * https://github.com/ultralytics/yolov5
 * https://github.com/nwojke/cosine_metric_learning/
 * https://github.com/nwojke/deep_sort
 * https://github.com/abhyantrika/nanonets_object_tracking
