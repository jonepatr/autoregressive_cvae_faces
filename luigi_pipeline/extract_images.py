import os
import shutil
import tempfile

import cv2

from luigi.util import requires
from luigi_pipeline import SuperTask
from luigi_pipeline.youtube_downloader import DownloadYoutubeVideo


@requires(DownloadYoutubeVideo)
class ExtractImages(SuperTask):
    def output(self):
        return self.get_output_path(self.yt_video_id)

    def run(self):
        cap = cv2.VideoCapture(self.input().path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = cap.read()
        count = 0
        with tempfile.TemporaryDirectory() as tmpd:
            while success:
                cv2.imwrite(os.path.join(tmpd, f"{count:05}.jpg"), image)
                success, image = cap.read()
                count += 1
                if count % 100 == 0:
                    self.set_progress_percentage(round(count / video_length) * 100)
            self.set_progress_percentage(100)
            self.output().makedirs()
            shutil.copytree(tmpd, self.output().path)
