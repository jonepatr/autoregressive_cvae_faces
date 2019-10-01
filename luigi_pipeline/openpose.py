import os
import shutil
import tempfile

import docker
from luigi.util import requires
from luigi_pipeline import SuperTask
from luigi_pipeline.text2face_docker_task import DockerTask
from luigi_pipeline.youtube_downloader import DownloadYoutubeVideo


@requires(DownloadYoutubeVideo)
class Openpose(DockerTask, SuperTask):
    resources = {"gpu": 1}

    def output(self):
        return self.get_output_path(self.yt_video_id)

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

            command = f"build/examples/openpose/openpose.bin -video /input/{input_file_name} -write_json /output -display 0 -render_pose 0"
            client.containers.run(
                "openpose",
                command=command,
                volumes=volumes,
                runtime="nvidia",
                environment={"CUDA_VISIBLE_DEVICES": "0,1"},
            )

            # video_file_name = os.path.join(td, self.yt_video_id + ".avi")

            self.output().makedirs()
            shutil.copytree(td, self.output().path)
