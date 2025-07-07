# Model based deep learning 

Minimal repo for the “Learned RPCA” Model based deep learning course, Final project.  
Contains two **training** scripts in Python and two **testing/inference** scripts in MATLAB.

| Data type | Train script (Python) | Output after training | Test script (MATLAB) | Extra input for test |
|-----------|----------------------|-----------------------|----------------------|----------------------|
| **Synthetic** | `python/training_codes_F1.py` | `synthetic_params.mat`, `synthetic_demo_data.mat` | `matlab/LearnedRPCA_demo.m` | `synthetic_demo_data.mat` + `synthetic_params.mat` |
| **Real video** | `python/training_LRPCA_real_data_F2.py` | `real_video_params.mat` | `matlab/LearnedRPCA_demo.m` (same) | Real‐video clip from [KITWARE collection](https://data.kitware.com/#collection/56f56db28d777f753209ba9f/folder/56f570368d777f753209baac) + `real_video_params.mat` |

---

## Quick start

```bash
# 1. Python environment
python -m venv venv && source venv/bin/activate          # or use conda
pip install -r requirements.txt

# 2. Train on synthetic data
python python/training_codes_F1.py        # saves *.mat files for MATLAB test

# 3. Train on real video
python python/training_LRPCA_real_data_F2.py \
    --video_path path/to/real_video.mp4   # see dataset link above
