# Model based deep learning 

Minimal repo for the “Learned RPCA” Model based deep learning course, Final project.  
Contains two **training** scripts in Python and two **testing/inference** scripts in MATLAB.

| Data type | Train script (Python) | Output after training | Test script (MATLAB) | Extra input for test |
|-----------|----------------------|-----------------------|----------------------|----------------------|
| **Synthetic** | `python/training_codes_synthetic_data.ipynb` | `synthetic_params.mat`, `synthetic_data.mat` | `matlab/testing_codes_matlab_model_based_F1.m` | `synthetic_data.mat` + `synthetic_params.mat` |
| **Real video** | `python/training_LRPCA_real_data.ipynb` | `real_video_params.mat` | `matlab/tasting_model_based_real_data.m` | Real‐video clip from [KITWARE collection](https://data.kitware.com/#collection/56f56db28d777f753209ba9f/folder/56f570368d777f753209baac) + `real_video_params.mat` |

---

## Quick start

```bash
# 1. Python environment
python -m venv venv && source venv/bin/activate    # or use conda
pip install -r requirements.txt

# === Synthetic data ===
# (choose one)

# 2A. Notebook (interactive)
jupyter notebook training_codes_synthetic_data.ipynb

# 2B. Script (non-interactive)
python python/training_codes_F1.py        # saves synthetic_params.mat + data

# === Real video ===
# (choose one)

# 4A. Notebook (interactive)
jupyter notebook training_LRPCA_real_data.ipynb         # prompts for video

# 4B. Script (non-interactive)
python python/training_LRPCA_real_data_F2.py \
    --video_path path/to/real_video.mp4                 # dataset link above

