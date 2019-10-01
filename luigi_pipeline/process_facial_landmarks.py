import json
from collections import Counter

import numpy as np

import luigi
from luigi_pipeline.dlib_facial_landmarks import DlibFacialLandmarks



class ProcessFacialLandmarks(luigi.Task):
    yt_video_id = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f'data/processed_facial_landmarls/{self.yt_video_id}.npy')

    def requires(self):
        return DlibFacialLandmarks()

    def complete(self):
        if self.output().exists():
            return True

        data = json.load(self.input().path)
        counter = Counter()
        for frame in data:
            counter[len(frame)] += 1
        
        if set(counter.keys()) - set([0,1]):
            return True

        return False

    def run(self):
        data = json.load(self.input().path)
        counter = Counter()
        for frame in data:
            counter[len(frame)] += 1
        
            frame_data = np.empty((0, 137))
            for i, frame in enumerate(data):
                if len(frame) == 1:
                    np.vstack([frame_data, frame[0]])
        

        with self.output().open('w') as f:
            np.save(, frame_data)
