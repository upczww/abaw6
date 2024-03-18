import os
import random
import numpy as np
import torch
import yaml
from munch import DefaultMunch
from tqdm import tqdm
import pickle
from model import Model
from utils.loss import *
import sys

if __name__ == "__main__":
    device = torch.device("cuda")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    config_path = "config/config.yml"
    yaml_dict = yaml.load(
        open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    cfg = DefaultMunch.fromDict(yaml_dict)

    pretrained_path = sys.argv[1]

    save_path = sys.argv[2]

    model = Model(cfg)

    print("loading from:", pretrained_path)
    pretrain_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    pretrain_dict = {k.replace("module.", ""): v for k, v in pretrain_dict.items()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    model = model.cuda()
    model.eval()

    window = 300
    stride = 300
    video_frames_dict = {}

    # 获取每个视频帧数
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
    print("videos:", len(video_frames_dict))

    # 获取视频列表
    test_label = "/data3/zww/abaw/testset/names_of_videos_in_each_test_set/Action_Unit_Detection_Challenge_test_set_release.txt"
    test_videos = []
    with open(test_label, "r") as f:
        for line in f:
            test_videos.append(line.strip())

    src_dir = "/data1/zww/abaw/AU_samples_testset"

    with open(save_path, "w") as f:
        f.write(
            "image_location,AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26\n"
        )
        for test_video in tqdm(test_videos):
            preds = []
            data = np.load(os.path.join(src_dir, test_video + ".npz"))
            select_mae_features = data["select_mae_features"].astype(np.float32)

            length = len(select_mae_features)

            for i in range(length // stride + 1):
                begin = stride * i
                if begin >= length:
                    break
                end = min(begin + window, length)

                x = select_mae_features[begin:end]

                x = torch.tensor(x).to(device).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(x)

                preds += (
                    torch.round(torch.sigmoid(outputs))
                    .int()
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

            frames_length = video_frames_dict[test_video]

            common_length = min(length, frames_length)

            # 填充共同部分
            for i in range(frames_length):
                write_str = ",".join(list(map(str, preds[i])))
                f.write(
                    test_video + "/" + str(i + 1).zfill(5) + ".jpg," + write_str + "\n"
                )
