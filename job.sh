#!/bin/bash
#SBATCH --time=00:180:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=32000M
#SBATCH --job-name=objectDetection
#SBATCH --output=./errors/error_%j.err
#SBATCH --error=./errors/error_%j.err
cd /home/abhijoymandal/abhijoy_working_dir/Segment-Anything-U-Specify
module load python pytorch
module load libglvnd
source ./venv/bin/activate
python3 main.py --input_image_path ./data/test_images/2023-07-13_RE_HTE_024_acetomi_solid_IPA_light_bottom.mkv --text vial

