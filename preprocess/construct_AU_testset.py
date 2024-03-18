import os
import pickle

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.interpolate import interp2d
from tqdm import tqdm

video_frames_dict = {}
with open(
    "/data3/zww/abaw/testset/test_set_examples/CVPR_6th_ABAW_AU_test_set_example.txt",
    "r",
) as f:
    lines = f.readlines()
    for line in lines[1:]:
        t = line.split("/")[0]
        if t not in video_frames_dict:
            video_frames_dict[t] = 1
        else:
            video_frames_dict[t] += 1


test_label = "/data3/zww/abaw/testset/names_of_videos_in_each_test_set/Action_Unit_Detection_Challenge_test_set_release.txt"

video_dir = "video"
features_dir = "/data1/zww/abaw/features"


sample_root_dir = "/data1/zww/abaw/AU_samples_testset"

mae_dir = os.path.join(features_dir, "mae")

os.makedirs(sample_root_dir, exist_ok=True)

video_files = []
with open(test_label, "r") as f:
    for line in f:
        video_files.append(line.strip())


for video_file in tqdm(video_files):

    len_label_frames = video_frames_dict[video_file]

    # 读取图片特征
    img_mae_feature_dir = os.path.join(mae_dir, video_file.split(".")[0])
    img_mae_features = np.zeros((len_label_frames, 768), dtype=np.float32)
    for i in tqdm(range(len_label_frames)):
        # 注意我们的帧数从0开始，保存的帧数从1开始
        img_mae_feature_file = os.path.join(
            img_mae_feature_dir, str(i + 1).zfill(5) + ".npy"
        )
        if os.path.exists(img_mae_feature_file):
            img_mae_feature = np.load(img_mae_feature_file)[0]
            img_mae_features[i] = img_mae_feature
        else:
            print(f"{i} not exist")

    print("img_mae_features", img_mae_features.shape)

    select_mae_features = img_mae_features
    print("select_mae_features", select_mae_features.shape)

    save_path = os.path.join(sample_root_dir, video_file.split(".")[0] + ".npz")

    np.savez(save_path, select_mae_features=select_mae_features)

    print("-" * 20)
