# HeinSight3.0



code and models: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11053915.svg)](https://doi.org/10.5281/zenodo.11053915)

dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11053823.svg)](https://doi.org/10.5281/zenodo.11053823)

## Installation
Clone `yolov5`
```
git clone https://github.com/ultralytics/yolov5
```
Install python packages via commands:
```
conda create -n hte python=3.9
conda activate hte
cd yolov5
pip3 install -r requirements.txt
cd ..
pip3 install -r requirements.txt
```

## Usage (images/videos):
Place the image/video in ./data/test_images.
Then from root run
```
python3 main.py --input_image_path <path of video> --nms_iou 0.1 --conf 0.2 --batch_size 64 --create_plots

```
Example
```
python3 main.py --input_image_path ./data/test_images/hetro_homo_cap_flash.mp4 --nms_iou 0.1 --conf 0.2 --batch_size 64 --create_plots
```

## Usage (camera):
From root run
```
python3 main.py --use_cameras <num_cameras> --nms_iou 0.1 --conf 0.2 --batch_size 64 --create_plots
```
<num_cameras> represents the number of cameras that you want to analyze

Example
If you want to analyze 2 cameras
```
python3 main.py --use_cameras 2 --nms_iou 0.1 --conf 0.2 --batch_size 64 --create_plots
```
To finish camera detections gracefully, press and hold the esc key to stop the analysis


## Publication:
https://chemrxiv.org/engage/chemrxiv/article-details/65e5481f9138d231619c1879
