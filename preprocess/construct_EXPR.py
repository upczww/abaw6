# %%
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.interpolate import interp2d
from tqdm import tqdm

label_root_dir = "/data1/zww/abaw/label/EXPR_Classification_Challenge/"
video_dir = "video"
features_dir = "/data1/zww/abaw/features"
crop_dir = "/data1/zww/abaw/crop"
sample_root_dir = "/data1/zww/abaw/EXPR_samples1"


mae_dir = os.path.join(features_dir, "mae")

mods = os.listdir(label_root_dir)

for mod in mods:
    label_dir = os.path.join(label_root_dir, mod)
    label_files = os.listdir(label_dir)

    sample_dir = os.path.join(sample_root_dir, mod)
    os.makedirs(sample_dir, exist_ok=True)

    for label_file in tqdm(label_files):
        video_path = os.path.join(
            video_dir,
            label_file.replace("_left", "")
            .replace("_right", "")
            .replace(".txt", ".mp4"),
        )
        if not os.path.exists(video_path):
            video_path = os.path.join(
                video_dir,
                label_file.replace("_left", "")
                .replace("_right", "")
                .replace(".txt", ".avi"),
            )
        video = cv2.VideoCapture(video_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("num_frames", num_frames)

        # 读取图片范围
        img_dir = os.path.join(crop_dir, label_file.split(".")[0])
        img_names = os.listdir(img_dir)
        img_names.sort()
        begin = int(img_names[0].split(".")[0]) - 1
        end = int(img_names[-1].split(".")[0]) - 1
        end = min(end, num_frames)

        # 读取标签
        labels = []
        with open(os.path.join(label_dir, label_file)) as f:
            lines = f.readlines()
            for line in lines[1:]:
                label = int(line.strip())
                labels.append(label)

        # 读取图片特征
        # mae
        img_mae_feature_dir = os.path.join(mae_dir, label_file.split(".")[0])
        prev_mae_feature = None
        img_mae_features = []
        for i in tqdm(range(begin, end)):
            # 注意我们的帧数从0开始，保存的帧数从1开始
            img_mae_feature_file = os.path.join(
                img_mae_feature_dir, str(i + 1).zfill(5) + ".npy"
            )
            if os.path.exists(img_mae_feature_file):
                img_mae_feature = np.load(img_mae_feature_file)
                prev_mae_feature = img_mae_feature
            else:
                img_mae_feature = prev_mae_feature
            img_mae_features.append(img_mae_feature)
        img_mae_features = np.stack(img_mae_features)  # 从begin到end的所有帧的特征
        print("img_mae_features", img_mae_features.shape)

        print("begin,end", begin, end)

        select_mae_features = img_mae_features
        print("select_mae_features", select_mae_features.shape)

        select_labels = np.array(labels[begin:end])

        print("select_labels", len(select_labels))

        save_path = os.path.join(sample_dir, label_file.split(".")[0] + ".npz")

        np.savez(
            save_path,
            select_mae_features=select_mae_features,
            select_labels=select_labels,
        )

        print("-" * 20)
