#!/bin/bash
#SBATCH --time=00:60:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32000M
#SBATCH --job-name=objectDetection
#SBATCH --output=./logs/output_%j.log
#SBATCH --error=./errors/error_%j.err
while getopts i:t flag
do
    case "${flag}" in
        i) image_path=${OPTARG};;
        t) text=${OPTARG};;
    esac
done
cd /home/abhijoymandal/abhijoy_working_dir/Segment-Anything-U-Specify
module load python pytorch
module load libglvnd
source ./venv/bin/activate
python3 py main.py --input_image_path $image_path --text $text