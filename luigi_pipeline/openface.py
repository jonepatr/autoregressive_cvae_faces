import os
import shutil
import subprocess
import tempfile

import docker
from luigi.util import requires
from luigi_pipeline import SuperTask
from luigi_pipeline.youtube_downloader import DownloadYoutubeVideo


@requires(DownloadYoutubeVideo)
class Openface(SuperTask):
    def output(self):
        return self.get_output_path(self.yt_video_id + "/" + self.yt_video_id + ".csv")

    def run(self):
        with tempfile.TemporaryDirectory() as td:
            volumes = {
                td: {"bind": "/output", "mode": "rw"},
                os.path.dirname(os.path.abspath(self.input().path)): {
                    "bind": "/input",
                    "mode": "ro",
                },
            }
            client = docker.from_env()

            input_file_name = os.path.basename(self.input().path)
            command = f"./build/bin/FeatureExtraction -f /input/{input_file_name} -out_dir /output -tracked -2Dfp -3Dfp -pdmparams -pose -aus -gaze -q"

            client.containers.run("openface", command=command, volumes=volumes)

            csv_file_name = os.path.join(td, self.yt_video_id + ".csv")
            video_file_name = os.path.join(td, self.yt_video_id + ".avi")

            self.output().makedirs()

            shutil.copy(video_file_name, self.output().path.replace(".csv", ".avi"))
            print(f"sed 's/, /,/g' {csv_file_name} > {self.output().path}")
            os.system(f"sed 's/, /,/g' {csv_file_name} > {self.output().path}")
