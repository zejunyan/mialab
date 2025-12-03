#!/bin/bash
#SBATCH --job-name="mia pipeline"
#SBATCH --account=gratis
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=epyc2

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate mia

# Run your script
python pipeline.py
