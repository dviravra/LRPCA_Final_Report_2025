clear; clc; close all;

%% ========== Load Real Video Data ==========
videoFile = "./processed_data/processed_video1.mp4";  
new_width = 320;    
new_height = 180;    
n_pixels = new_width*new_height;

vidObj = VideoReader(videoFile);
frames = {};
while hasFrame(vidObj)
    frame = readFrame(vidObj);
    % Resize
    frame = imresize(frame, [new_height, new_width]);
    % Convert to grayscale if needed
    if size(frame,3) == 3
        frame = rgb2gray(frame);
    end
    % Convert to double in [0,1]
    frame = im2double(frame);
    % Flatten each frame into a column
    frames{end+1} = frame(:);
end

Y = cell2mat(frames);  % size(Y) = (new_height*new_width, num_frames)
Y = sparse(Y);
[n, numFrames] = size(Y);
fprintf('Loaded video: %d pixels per frame, %d frames.\n', n, numFrames);
fprintf("========================================\n");

%% ========== Load the Learned Model Parameters ==========
model_path = "./trained_models/LRPCA_real_r2_2025_06_28_10-57.mat";
load(model_path, 'ths', 'step_U', 'step_V');  % step_V size [iter_num×batch_size×r] % step_U size [iter_num×n_pixels×r]

% new size
[~, numFrames] = size(Y);

%% ===== interpolate step_V =========
[T, origFrames, r] = size(step_V);
if origFrames ~= numFrames
    origX = 1:origFrames;                             
    newX  = linspace(1, origFrames, numFrames);       

    newStepV = zeros(T, numFrames, r);
    for t = 1:T
        for k = 1:r
            % vector step_V for layer t and rank r
            v = squeeze(step_V(t,:,k));               %  [1×origFrames]
            %  interpolation for numFrames
            newStepV(t,:,k) = interp1(origX, v, newX, 'linear', 'extrap');
        end
    end

    step_V = newStepV;  % new size [iter_num×numFrames×r]
end

%% parameters
r = 2;  % rank
zeta = ths * (n_pixels/35000) * (r/2) ;
zeta = double(zeta(:));         % [iter_num×1]
etaU = double(step_U);          % [iter_num×n_pixels×r]
etaV = double(step_V);          % [iter_num×numFrames×r]

%% ========== Running LRPCA with Matrix Step Sizes ==========
[X, L, R, finalErr] = LearnedRPCA_real_matrix(Y, r, zeta, etaU, etaV);

fprintf('Final relative reconstruction error (||Y - X||/||Y||): %e\n', finalErr);
% ============  (MSE/RMSE/MAE) ===============
E = full(Y) - X;                        
[n, numFrames] = size(Y);               
totalPixels = n * numFrames;

MSE  = mean( E(:).^2 );                % mean squared error per pixel
RMSE = sqrt( MSE );                    % root MSE
MAE  = mean( abs(E(:)) );              % mean absolute error

fprintf('MSE per pixel: %e\n'  , MSE);
fprintf('RMSE per pixel: %e\n' , RMSE);
fprintf('MAE per pixel:  %e\n' , MAE);

%% ========== Display a Sample Frame ==========
frame_idx = round(numFrames/2);  
X_full = full(X);                
recovered_background = X_full(:, frame_idx);
sparse_component = full(Y(:, frame_idx)) - recovered_background;
original_frame = full(Y(:, frame_idx));

% Reshape to 2D images
original_frame = reshape(original_frame, [new_height, new_width]);
recovered_background = reshape(recovered_background, [new_height, new_width]);
sparse_component = reshape(sparse_component, [new_height, new_width]);

figure('Name','Visual Comparison');
subplot(1,3,1);
imshow(original_frame, []);
title('Original Frame');

subplot(1,3,2);
imshow(recovered_background, []);
title('Recovered Background');

subplot(1,3,3);

lowVal = prctile(sparse_component(:), 98);
highVal = prctile(sparse_component(:), 0);
sparse_vis = mat2gray(sparse_component, [lowVal, highVal]);
imshow(sparse_vis);
title('Sparse Component');

%% ========== Implementation of LearnedRPCA for Real Data ==========
function [X, L, R, finalError] = LearnedRPCA_real_matrix(Y, r, zeta, etaU, etaV)
    [n_pixels, numFrames] = size(Y);    % n_pixels, numFrames
    T      = length(zeta);
    time_counter = 0;

    % ---- Initialization ----
    tStart = tic;
    S0 = Thre(full(Y), zeta(1));
    [U0,S0d,V0] = svds(Y-S0, r);
    L = U0 * sqrt(S0d);
    R = V0 * sqrt(S0d);
    time_counter = time_counter + toc(tStart);

    % ---- Unrolled updates ----
    for t = 1:(T-1)
        tStart = tic;
        X  = L * R';
        S  = Thre(full(Y - X), zeta(t+1));

        % Compute "gradients"
        L_plus = ((X + S - Y) * R) / (R' * R + eps*eye(r));    % [n_pixels × r]
        R_plus = ((X + S - Y)' * L) / (L' * L + eps*eye(r));    % [n_pixels × r]

        % Load learned step‐sizes
        stepU    = squeeze( etaU(t+1, :, :) );   % -> [n_pixels × r]
        stepV = squeeze( etaV(t+1, :, :) ); % -> [numFrames × r]
        % Debug
        assert(isequal(size(R_plus), size(stepV)), 'size mismatch: R_plus vs. stepV');
        % Element‐wise update
        L = L - L_plus .* stepU;                     
        R = R - R_plus .* stepV;                   
        time_counter = time_counter + toc(tStart);
        rec_err = norm(Y - L*R','fro')/norm(Y,'fro');
        fprintf("Layer %d, RelError: %e, Time: %f\n", t, rec_err, time_counter);
    end
    fprintf("========================================\n");
    X = L * R';
    finalError = norm(Y - X, 'fro') / norm(Y, 'fro');
end


%% ========== Soft Thresholding Function ==========
function S = Thre(S, theta)
    S = sign(S) .* max(abs(S) - theta, 0.0);
end
