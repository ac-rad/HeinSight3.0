#!/bin/bash
#SBATCH --time=00:180:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32000M
#SBATCH --job-name=objectDetection
#SBATCH --output=./errors/error_%j.err
#SBATCH --error=./errors/error_%j.err
cd /home/abhijoymandal/abhijoy_working_dir/Segment-Anything-U-Specify/
module load python pytorch
module load libglvnd
source ./venv/bin/activate
python3 ./yolo_main.py --input_image_path ./data/test_images/dimmed_resized_2023-06-29_RE-HTE-UoT_test_004.mp4 --batch_size 32

