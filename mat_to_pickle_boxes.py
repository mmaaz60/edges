import os
import pickle
from os.path import dirname, join as pjoin
import scipy.io as sio
import cv2
import numpy as np
import torch


mat_dir = "./edge_boxes"
out_dir = "./edge_boxes_ore"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for file in os.listdir(mat_dir):
    file_path = f"{mat_dir}/{file}"
    mat_contents = sio.loadmat(file_path)
    mat_boxes = mat_contents['bbox']
    with open(f"{out_dir}/{file.split('.')[0]}.pkl", "wb") as f:
        pickle.dump(torch.tensor(mat_boxes), f)
print("")
