import os
import re

import luigi


class Params(luigi.Config):
    yt_video_id = luigi.Parameter()
    data_dir = luigi.Parameter()


class SuperTask(luigi.Task):
    def get_output_path(self, file_name):
        processing_details = re.sub(
            rf"\(|\)|{self.task_family}|yt_video_id=.+?(, |\))|data_dir=.+?(, |\))",
            "",
            repr(self),
        )
        processing_details = processing_details.replace(", ", "-").replace("=", "_")

        path = os.path.join(
            self.data_dir, self.task_family, processing_details, file_name
        )
        return luigi.LocalTarget(path)
