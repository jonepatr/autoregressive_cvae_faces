import bz2
import os
import shutil
import urllib.request

import luigi
from luigi.util import inherits
from luigi_pipeline import Params


class DownloadFile(luigi.Task):
    data_dir = luigi.Parameter()
    url = luigi.Parameter()
    compression = luigi.Parameter("")

    def output(self):
        filename = os.path.basename(self.url)
        return luigi.LocalTarget(
            os.path.join(self.data_dir, "files", filename), format=luigi.format.Nop
        )

    def run(self):
        print(self.url)
        with urllib.request.urlopen(self.url) as response:
            if not self.compression:
                with self.output().open("wb") as f:
                    self.output().makedirs()
                    shutil.copyfileobj(response, f)
            elif self.compression == "bz2":
                decompressor = bz2.BZ2Decompressor()
                data = decompressor.decompress(response.read())
                self.output().makedirs()
                with self.output().open("wb") as f:
                    f.write(data)
