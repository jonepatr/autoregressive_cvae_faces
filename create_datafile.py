import glob
import os
import random
import json


openpose_dir = "/data_dir/PostProcessOpenpose/ffmpeg_bin_ffmpeg-fps_30.0-will_interpolate_True-min_confidence_0.7-min_value_0.001-min_mean_0.1-max_mean_1.0-min_std_0.001/*"
files = [os.path.basename(x).replace(".npy", "") for x in glob.glob(openpose_dir)]
random.shuffle(files)

with open("datafiles.json", "w") as f:
    json.dump(
        {
            "train": files[: round(len(files) * 0.8)],
            "val": files[round(len(files) * 0.8) : round(len(files) * 0.9)],
            "test": files[round(len(files) * 0.9) :],
        },
        f,
    )
