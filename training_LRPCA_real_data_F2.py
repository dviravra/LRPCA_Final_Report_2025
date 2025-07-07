import os
import math
import random
import time
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==================== Configuration ====================
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype     = torch.float32
r               = 2        # target rank 
maxIt           = 8        # number of unrolled layers 
frame_skip      = 2        # sample frame 
resize_factor   = 1.0     # downsample 
batch_size      = 16       
num_videos      = 10        # number of training videos 
val_videos_num  = 1        # number of validation videos
ths_initial     = 0.5     # initial threshold 
step_initial    = 0.8      # initial step size 
lr_ths          = 1e-5     
lr_step         = 1e-2     
Nepoches_pre    = 10       
Nepoches_full   = 20       
loss_list       = []       
lambda_s        = 0.14     # sparsity penalty coefficient

# ================= Video Loader (systematic sampling) =================
def load_video_batches(path):
    """Load a video, convert to grayscale, downsample, flatten frames, and chunk into batches."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            # Convert to grayscale and downsample 
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float32) / 255.0  # normalize 
            frames.append(frame.flatten())
        idx += 1
    cap.release()
    if len(frames) == 0:
        return []  
    # Stack frames into matrix Y of shape (n_pixels, T_frames)
    Y = np.stack(frames, axis=1)  
    Y_t = torch.from_numpy(Y).to(device)
    chunks = torch.chunk(Y_t, math.ceil(Y_t.shape[1] / batch_size), dim=1)
    return [c for c in chunks if c.shape[1] >= r]

# ==================== MatNet (Unrolled LRPCA Model) ====================
def loss_fn(Y, X_rec, S, lambda_s=0.05):
    # Sparsity penalty (normalized L1 of S)
    sparse_pen = torch.norm(S, p=1) / S.numel()
    rec_err = torch.norm(Y - X_rec, p='fro') / (torch.norm(Y, p='fro') + 1e-8) + lambda_s * sparse_pen
    return rec_err 

class MatNet(nn.Module):
    def __init__(self, n_pixels, rank,batch_size):
        """Unrolled LRPCA network with per-layer thresholds."""
        super(MatNet, self).__init__()
        # Learnable soft-thresholds
        self.ths_v = nn.ParameterList([
            nn.Parameter(torch.tensor(ths_initial, dtype=dtype, device=device), requires_grad=True)
            for _ in range(maxIt)
        ])
        # Learnable step sizes 
        self.step_U = nn.ParameterList([
            nn.Parameter(torch.full((n_pixels, rank), step_initial, dtype=dtype, device=device), requires_grad=True)
            for _ in range(maxIt)
        ])
        # Learnable step sizes for V 
        self.step_V = nn.ParameterList([
            nn.Parameter(torch.full((batch_size,rank), step_initial, dtype=dtype, device=device), requires_grad=True)
            for _ in range(maxIt)
        ])
        self.ths_v[0].requires_grad = False
        self.step_U[0].requires_grad = False
        self.step_V[0].requires_grad = False

    def thre(self, X, tau):
        return torch.sign(X) * torch.clamp(torch.abs(X) - tau, min=0.0)

    def forward(self, Y, num_layers):
        """
        Forward pass through num_layers of the unrolled LRPCA.
        Y is an (n_pixels × T) matrix (T = number of frames in batch).
        Returns the reconstruction loss for training.
        """
        current_batch = Y.shape[1]
        # Initial
        S = self.thre(Y, self.ths_v[0])       
        X0 = Y - S                            
        try:
            U0, S0, V0 = torch.svd_lowrank(X0, q=r, niter=4)  
        except RuntimeError:
            # Fallback to full SVD on CPU if low-rank SVD fails 
            U_cpu, S_cpu, V_cpu = torch.linalg.svd(X0.cpu(), full_matrices=False)
            U0 = U_cpu[:, :r].to(device)
            S0 = S_cpu[:r].to(device)
            V0 = V_cpu[:r, :].t().to(device)
        sqrtS = torch.sqrt(S0)
        U = U0 * sqrtS.unsqueeze(0)  # U: (n_pixels × r)
        V = V0 * sqrtS.unsqueeze(0)  # V: (T × r)
        eps = 1e-2  # regularization 
        for t in range(1, num_layers):
            X = U @ V.t()
            E = Y - X
            S = self.thre(E, self.ths_v[t])
            E = E - S # Update new X
            # Compute update steps for U and V
            VtV = V.t() @ V + eps * torch.eye(r, device=device)
            UtU = U.t() @ U + eps * torch.eye(r, device=device)
            Vkernel = torch.inverse(VtV)   # (r × r)
            Ukernel = torch.inverse(UtU)   # (r × r)
            dU = (E @ V) @ Vkernel        
            dV = (E.t() @ U) @ Ukernel    
            U = U + dU * self.step_U[t]   
            V = V + dV * self.step_V[t][:current_batch,:]   
        X_rec = U @ V.t()
        loss = loss_fn(Y, X_rec, S, lambda_s=0.05)
        return loss

    def EnableSingleLayer(self, l):
        for i in range(maxIt):
            self.ths_v[i].requires_grad = False
            self.step_U[i].requires_grad = False
            self.step_V[i].requires_grad = False
        if l > 0:  # layer 0 remains fixed
            self.ths_v[l].requires_grad = True
            self.step_U[l].requires_grad = True
            self.step_V[l].requires_grad = True

    def EnableLayers(self, L):
        for i in range(maxIt):
            if i == 0:
                self.ths_v[i].requires_grad = False
                self.step_U[i].requires_grad = False
                self.step_V[i].requires_grad = False
            else:
                self.ths_v[i].requires_grad = (i < L)
                self.step_U[i].requires_grad = (i < L)
                self.step_V[i].requires_grad = (i < L)

# ================ Prepare data (load videos into batches) ================
folder = "/content/my_local_videos"
all_videos = [os.path.join(folder, f) for f in os.listdir(folder)
              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
all_videos.sort()
random.shuffle(all_videos)
train_videos = all_videos[:num_videos]
val_videos   = all_videos[num_videos : num_videos + val_videos_num]
print("Train videos:", train_videos)
print("Val videos:  ", val_videos)

# Load all training and validation video frames into batches
train_batches = []
for vid in train_videos:
    train_batches += load_video_batches(vid)
val_batches = []
for vid in val_videos:
    val_batches += load_video_batches(vid)
print(f"Total training batches: {len(train_batches)}, Total validation batches: {len(val_batches)}")

# Ensure all batches have consistent spatial dimension
if len(train_batches) == 0:
    raise RuntimeError("No training frames loaded. Check video paths or frame_skip settings.")
n_pixels = train_batches[0].shape[0]
for b in train_batches + val_batches:
    assert b.shape[0] == n_pixels, "Inconsistent frame size across videos. Please resize videos to same resolution."

# ==================== Initialize Model and Optimizers ====================
net = MatNet(n_pixels, r,batch_size).to(device)
optimizers = [None] * maxIt
schedulers = [None] * maxIt
for t in range(1, maxIt):
    opt = optim.Adam([
        {'params': [net.ths_v[t]],               'lr': lr_ths},
        {'params': [net.step_U[t], net.step_V[t]], 'lr': lr_step}
    ], weight_decay=1e-6)
    optimizers[t] = opt
    schedulers[t] = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)

train_hist, val_hist = [], []
start_time = time.time()

# ================ Training: Layer-wise (pre-training) ================
for layer in range(1, maxIt):
    print(f"\n================ Layer {layer} Pre-train ================")
    net.train()            
    net.EnableSingleLayer(layer)
    if layer > 0:
       with torch.no_grad():
           net.ths_v[layer].data.copy_(net.ths_v[layer-1].data * 0.1)
           init_th = net.ths_v[layer].item()
    for ep in range(Nepoches_pre):
        random.shuffle(train_batches)
        total_loss, total_frames = 0.0, 0
        # Only train parameters of the current layer
        net.EnableSingleLayer(layer)
        for Y_batch in train_batches:
            optimizers[layer].zero_grad()
            loss = net(Y_batch, layer + 1)  # forward 
            if torch.isnan(loss):
                print("  NaN loss encountered, skipping this batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizers[layer].step()
            # Clamp threshold of this layer to [0, 0.5] to prevent negative or overly large values
            with torch.no_grad():
                net.ths_v[layer].clamp_(0.0, 0.5)
                net.step_U[layer].data.clamp_( 0.0, 1.0)
                net.step_V[layer].data.clamp_(0.0, 1.0)
            total_loss   += loss.item() * Y_batch.shape[1]
            total_frames += Y_batch.shape[1]
        train_avg = total_loss / (total_frames if total_frames > 0 else 1)
        # Validation 
        net.eval()
        val_loss_sum, val_frames = 0.0, 0
        with torch.no_grad():
            for Y_batch in val_batches:
                l = net(Y_batch, layer + 1)
                val_loss_sum += l.item() * Y_batch.shape[1]
                val_frames   += Y_batch.shape[1]
        val_avg = val_loss_sum / (val_frames if val_frames > 0 else 1)
        schedulers[layer].step(val_avg)
        train_hist.append(train_avg)
        val_hist.append(val_avg)
        # Debug print
        th_val = net.ths_v[layer].item()
        stepU_mat = net.step_U[layer].data
        stepV_vec = net.step_V[layer].data
        etaU_norm = torch.norm(stepU_mat).item()
        etaU_min  = torch.min(stepU_mat).item()
        etaU_max  = torch.max(stepU_mat).item()
        etaV_norm = torch.norm(stepV_vec).item()
        etaV_min  = torch.min(stepV_vec).item()
        etaV_max  = torch.max(stepV_vec).item()
        print(f" Epoch {ep+1:2d}: train_loss={train_avg:.4f}, val_loss={val_avg:.4f}, "
              f"ths[{layer}]={th_val:.4f}, "
              f"etaU[{layer}] (norm={etaU_norm:.4f}, min={etaU_min:.4f}, max={etaU_max:.4f}), "
              f"etaV[{layer}] (norm={etaV_norm:.4f}, min={etaV_min:.4f}, max={etaV_max:.4f})")

    # ================ Full fine-tuning up to current layer ================
    print(f"================ Layer {layer} Full-train ================")
    for ep in range(Nepoches_full):
        random.shuffle(train_batches)
        net.train()
        total_loss, total_frames = 0.0, 0
        # Enable training for all layers up to `layer`
        net.EnableLayers(layer + 1)
        for Y_batch in train_batches:
            for t in range(1, layer + 1):
                optimizers[t].zero_grad()
            loss = net(Y_batch, layer + 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            for t in range(1, layer + 1):
                optimizers[t].step()
            total_loss   += loss.item() * Y_batch.shape[1]
            total_frames += Y_batch.shape[1]
        train_avg = total_loss / (total_frames if total_frames > 0 else 1)
        train_hist.append(train_avg)
        if (ep + 1) % 10 == 0 or ep == Nepoches_full - 1:
            print(f" Epoch {ep+1:2d}: combined_train_loss={train_avg:.4f}")

# =================== Training Complete, Save Results ===================
elapsed = time.time() - start_time
print(f"\nTraining completed in {elapsed:.1f} seconds.")

# Prepare results for saving
ths_res   = [param.detach().cpu().item() for param in net.ths_v]  # list of threshold values
stepU_res = [param.detach().cpu().numpy() for param in net.step_U]  # list of (n_pixels×r) arrays
stepV_res = [param.detach().cpu().numpy() for param in net.step_V]  # list of (r,) arrays
stepU_arr = np.stack(stepU_res, axis=0)
stepV_arr = np.stack(stepV_res, axis=0)
results_mat = {
    'ths': np.array(ths_res),
    'step_U': stepU_arr,
    'step_V': stepV_arr,
    'train_loss': np.array(train_hist),
    'val_loss': np.array(val_hist)
}
out_path = f"LRPCA_real_r{r}_{datetime.now():%Y_%m_%d_%H-%M}.mat"
sio.savemat(out_path, results_mat)
print("Saved learned parameters to", out_path)

# If Google Drive is mounted, also save a copy to drive
drive_save_path = os.path.join("/content/drive/MyDrive", os.path.basename(out_path))
try:
    sio.savemat(drive_save_path, results_mat)
    print("Saved results to Drive:", drive_save_path)
except Exception as e:
    pass

# ==================== Plot training curve ====================
plt.figure()
plt.plot(train_hist, label='Train Loss')
if len(val_hist) > 0:
    plt.plot(val_hist, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (average per epoch)')
plt.legend()
plt.title('LRPCA Training Loss Curve')
plt.show()
