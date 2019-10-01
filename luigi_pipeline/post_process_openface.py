import numpy as np
import pandas as pd

import luigi
from luigi.util import requires
from luigi_pipeline import SuperTask
from luigi_pipeline.openface import Openface
from luigi_pipeline.post_process_openpose import PostProcessOpenpose

COLUMNS = [
    "AU01_r",  # Inner brow raiser
    "AU02_r",  # Outer brow raiser
    "AU04_r",  # Brow lowerer
    "AU45_r",  # Brow lowerer
    "pose_Rx",  # head rotation
    "pose_Ry",  # head rotation
    "pose_Rz",  # head rotation
    # "gaze_angle_x",  # eye rotation
    # "gaze_angle_y",  # eye rotation
]


@requires(Openface, PostProcessOpenpose)
class PostProcessOpenface(SuperTask):
    with_blinks = luigi.BoolParameter(True)

    def output(self):
        return self.get_output_path(self.yt_video_id + ".npy")

    def run(self):
        openface_input, post_processed_openpose_input = self.input()

        df = pd.read_csv(openface_input.path, delimiter=", ")
        openpose_data = np.load(post_processed_openpose_input.path, allow_pickle=True)

        # determine whether frame is good or not

        # 1 use openpose filter
        data = {}

        for start_frame, frames in openpose_data.item().items():
            data[start_frame] = df.loc[start_frame : start_frame + len(frames) - 1][
                COLUMNS
            ].to_numpy()

        # 2  get dlib data -> look for frames containing multiple people
        # Might try this later

        self.output().makedirs()

        np.save(self.output().path, data)
