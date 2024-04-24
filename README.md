# HeinSight3.0



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11053915.svg)](https://doi.org/10.5281/zenodo.11053915)

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

## Usage:
Place the video in ./data/test_images.
Then from root run
```
python3 main.py --input_image_path <path of video> --nms_iou 0.1 --conf 0.2 --batch_size 64 --create_plots

```
Example
```
python3 main.py --input_image_path ./data/test_images/hetro_homo_cap_flash.mp4 --nms_iou 0.1 --conf 0.2 --batch_size 64 --create_plots
```

## Dataset:
Dataset and model statistics can be found at https://doi.org/10.5281/zenodo.11053823

## Publication:
https://chemrxiv.org/engage/chemrxiv/article-details/65e5481f9138d231619c1879
