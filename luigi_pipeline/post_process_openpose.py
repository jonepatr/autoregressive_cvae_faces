import json
import os
import re
from glob import glob
# import lazy_import
import numpy as np
import luigi
from luigi.util import requires
from luigi_pipeline import SuperTask
from luigi_pipeline.openpose import Openpose
from luigi_pipeline.youtube_downloader import DownloadYoutubeVideo
import cv2

#cv2 = lazy_import.lazy_callable("cv2")


@requires(Openpose, DownloadYoutubeVideo)
class PostProcessOpenpose(SuperTask):
    will_interpolate = luigi.BoolParameter(True)
    min_confidence = luigi.FloatParameter(0.7)
    min_value = luigi.FloatParameter(0.001)
    min_mean = luigi.FloatParameter(0.1)
    max_mean = luigi.FloatParameter(1.0)
    min_std = luigi.FloatParameter(0.001)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = None
        self.height = None

    def output(self):
        return self.get_output_path(self.yt_video_id + ".npy")

    def normalize_landmarks(self, points1, maxnorm=True):
        c1 = np.mean(points1, axis=0)
        points1 -= c1[None, :]  # unsqueeze
        s1 = np.std(points1, axis=0)
        points1 /= s1[None, :]  # unsqueeze
        if maxnorm:
            points1 /= np.abs(points1).max(axis=0)[None, :]
        return points1

    def get_valid_people(self, people):
        valid_people = []

        for person in people:
            d = np.array(person["face_keypoints_2d"]).reshape(-1, 3)
            norm_d = d / np.array([self.width, self.height, 1])[None, :]

            confidence = np.mean(d[:, -1])

            if np.isnan(d).any():
                continue

            if np.min(norm_d) < self.min_value:
                continue

            if confidence < self.min_confidence and confidence > 1:
                continue

            mean = np.mean(norm_d, axis=0)

            if np.min(mean) < self.min_mean or np.max(mean) > self.max_mean:
                continue

            if np.min(np.std(norm_d, axis=0)) < self.min_std:
                continue

            valid_people.append(self.normalize_landmarks(d[:, :-1]))

        return valid_people

    def read_fucked_json(self, path):
        with open(path) as f:
            p = f.read()
            frame = json.loads(
                re.sub(r"-?\binf\b", "Infinity", re.sub(r"-?\bnan\b", "NaN", p))
            )
        return frame

    def interpolate(self, frame_1, frame_2):
        return np.mean([frame_1, frame_2], axis=0)

    def run(self):
        input_openpose, input_video = self.input()

        video = cv2.VideoCapture(input_video.path)
        self.width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        list_of_frames = {}
        tmp_frames = []
        file_paths = sorted(glob(os.path.join(input_openpose.path, "*.json")))
        first_frame = None
        for n in range(len(file_paths) - 1):
            file_path = file_paths[n]

            frame = self.read_fucked_json(file_path)

            people = self.get_valid_people(frame["people"])
            people_count = len(people)

            if people_count == 1:
                if not first_frame:
                    first_frame = n
                tmp_frames.append(people[0])

            else:

                next_frame = self.read_fucked_json(file_paths[n + 1])
                people_in_next_frame = self.get_valid_people(next_frame["people"])

                if (
                    people_count == 0
                    and len(people_in_next_frame) == 1
                    and tmp_frames
                    and self.will_interpolate
                ):
                    tmp_frames.append(
                        self.interpolate(tmp_frames[-1], people_in_next_frame[0])
                    )
                    people_count = 1
                elif first_frame:
                    list_of_frames[first_frame] = np.array(tmp_frames)
                    tmp_frames = []
                    first_frame = None

        self.output().makedirs()
        np.save(self.output().path, list_of_frames)
