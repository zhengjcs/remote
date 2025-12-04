<!-- <a href="https://apps.apple.com/app/id1452689527" target="_blank">
<img src="https://user-images.githubusercontent.com/26833433/82944393-f7644d80-9f4f-11ea-8b87-1a5b04f555f1.jpg" width="1000"></a>
&nbsp

![CI CPU testing](https://github.com/zhengjcs/remote/blob/main/picture/P0065.png) -->
![image](picture/100000637.bmp)

The goal of the work described in this paper is to achieve a rotated bounding box detection of directional objects for remote-sensing images. Specifically, a separate information dimension about the object angle is introduced, along with the corresponding loss and regression functions. This method can well indicate the direction of the object to be detected. By determining the angle, the proposed bounding box shape has a more accurate width and height to fit the actual target, which is of great significance not only in intuitive visual experience, but also in the re-application of the result data. At the same time, most of the current remote-sensing datasets are labeled with four points (x1 y1, x2 y2, x3 y3, x4 y4) or the width and height of the object are not clearly defined. Therefore in order to obtain the desired effect after training, the data are reprocessed before training to calculate the true width and height of the object and the deflection angle relative to the coordinate axis. To sum up, the main contributions of this study are as follows.


## Model Checkpoints
DOTA Dataset
| Model | PL | BD | BR | GTF | SV | LV | SH | TC | BC | ST | SBF | RA | HA | SP | HC | mAP |
|---------- |------ |------ |------ |------ |------|------ |------|------ |------| ------| ------|------|------|------|------| :------: |
| [ROD-DOTA]   | 66.5 | 52.7 | 13.1 | 27.0 | 45.7 | 68.8 | 41.0 | 88.5 | 40.5 | 22.6 | 28.2 | 21.1 | 51.5 | 41.6 | 25.0 | 42.2 

HRSC2016
| Model | Backbone | Size> | mAP50 | 
|---------- |------ |------ | :------: |
| [ROD-HRSC]    | DarkNet53     | 800 × 800    | 96.2  


## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.6`. To install run:
```bash
$ pip install -r requirements.txt
```



