# %%
import os

import numpy as np
from tqdm import tqdm


# %%
def split_sample(length, window=30, stride=15):
    splits = []
    for i in range(length // stride):
        begin = stride * i
        end = begin + window
        if end > length:
            begin = length - window
            end = length
            break
        splits.append([begin, end])
    return splits


# %%
dest_dir = "/data1/zww/abaw/VA_splits_w300_s200"
src_dir = "/data1/zww/abaw/VA_samples1"
window = 300
stride = 200
mods = os.listdir(src_dir)
for mod in mods:
    save_dir = os.path.join(dest_dir, mod)
    os.makedirs(save_dir, exist_ok=True)
    samples_dir = os.path.join(src_dir, mod)
    samples = os.listdir(samples_dir)
    for sample in tqdm(samples):
        data = np.load(os.path.join(samples_dir, sample), allow_pickle=True)
        select_mae_features = data["select_mae_features"]

        select_valences = data["select_valences"]
        select_arousals = data["select_arousals"]
        length = len(select_mae_features)
        splits = split_sample(length, window, stride)
        for idx, s in enumerate(splits):
            b = s[0]
            e = s[1]
            mae_features = select_mae_features[b:e]

            valences = select_valences[b:e]
            arousals = select_arousals[b:e]

            if len(valences) != window:
                continue
            save_path = os.path.join(
                save_dir,
                os.path.basename(sample).split(".")[0]
                + "_"
                + str(idx).zfill(4)
                + ".npz",
            )
            np.savez(
                save_path,
                mae_features=mae_features,
                valences=valences,
                arousals=arousals,
            )
