from glob import glob
from tqdm import tqdm
import os
import subprocess


files = list(glob("/mhi_mimicry_data/Sessions/*/*FaceFar1*"))
bar = tqdm(total=len(files))
for file_ in files:

    file_30fps = os.path.join(
        os.path.dirname(file_).replace("Sessions", "Sessions_30fps"),
        os.path.basename(file_),
    )
    if os.path.exists(file_30fps):
        continue

    os.makedirs(
        os.path.dirname(file_).replace("Sessions", "Sessions_30fps"), exist_ok=True
    )
    base_name = os.path.basename(os.path.dirname(file_)) + "_" + os.path.basename(file_)
    bar.set_description_str(base_name)
    subprocess.Popen(
        [
            "/usr/bin/ffmpeg",
            "-y",
            "-i",
            file_,
            "-c:v",
            "libx264",
            "-r",
            "30",
            file_30fps,
        ],
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()

    bar.update(1)
