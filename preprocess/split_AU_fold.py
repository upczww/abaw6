import os
import numpy as np
import glob

from tqdm import tqdm
from sklearn.model_selection import KFold


def split_sample(length,window=30,stride=15):
    splits = []
    for i in range(length//stride):
        begin = stride * i
        end = begin + window
        if end > length:
            begin = length - window
            end = length
            break
        splits.append([begin,end])
    return splits

dest_dir = "/data1/zww/abaw/AU_splits_w300_s200"
src_dir = "/data1/zww/abaw/AU_samples1"
window = 300
stride = 200


all_samples = np.array(glob.glob(os.path.join(src_dir,"**/*.npz")))


kf = KFold(n_splits=4)

for idx,(train_samples,val_samples) in enumerate(kf.split(all_samples)):
    print("#",idx)
    fold_dir = os.path.join(dest_dir+"_fold_"+str(idx+1))

    modes_samples = {"Train_Set":train_samples,"Validation_Set":val_samples}

    for mode in modes_samples:
        save_dir = os.path.join(fold_dir,mode)
        print("save_dir:",save_dir)
        os.makedirs(save_dir,exist_ok=True)
        samples_idx = modes_samples[mode]
        samples = all_samples[samples_idx]
        for sample in tqdm(samples):
            data = np.load(sample, allow_pickle=True)
            select_mae_features = data["select_mae_features"]
            select_labels=data["select_labels"]

            length = len(select_mae_features)
            splits = split_sample(length,window,stride)
            for idx,s in enumerate(splits):
                b = s[0]
                e = s[1]
                mae_features = select_mae_features[b:e]

                labels = select_labels[b:e]

                if len(labels) != window:
                    continue
                save_path = os.path.join(save_dir,str(idx).zfill(4)+"_"+os.path.basename(sample))
                np.savez(
                    save_path,
                    mae_features=mae_features,
                    labels=labels,
                )


