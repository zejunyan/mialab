# Medical Image Analysis Laboratory

Welcome to the medical image analysis laboratory (MIALab).
This repository contains all code you will need to get started with classical medical image analysis.

During the MIALab you will work on the task of brain tissue segmentation from magnetic resonance (MR) images.
We have set up an entire pipeline to solve this task, specifically:

- Pre-processing
- Registration
- Feature extraction
- Voxel-wise tissue classification
- Post-processing
- Evaluation

After you complete the exercises, dive into the 
    
    pipeline.py 

script to learn how all of these steps work together. 

During the laboratory you will get to know the entire pipeline and investigate one of these pipeline elements in-depth.
You will get to know and to use various libraries and software tools needed in the daily life as biomedical engineer or researcher in the medical image analysis domain.

Enjoy!

----

Found a bug or do you have suggestions? Open an issue or better submit a pull request.


## Run on ubelix

To run on ubelix make create an environment:

```bash
# Load Anaconda module
module load Anaconda3

# Initialize conda for your shell
eval "$(conda shell.bash hook)"

# Create a new environment (replace dl_a2 with your preferred name)
conda create -n mia python=3.10 -y
```


```bash
# Activate your environment
conda activate mia

# Install all packages from requirements.txt
pip install -r requirements.txt
```
