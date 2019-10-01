import csv
import random
import shutil
import subprocess
import tempfile

import requests

import luigi
from bs4 import BeautifulSoup
from luigi.util import inherits
from luigi_pipeline import Params, SuperTask
from pytube import YouTube


@inherits(Params)
class YoutubeDownloader(SuperTask):
    ffmpeg_bin = luigi.Parameter("ffmpeg")
    def output(self):
        return self.get_output_path(f"{self.yt_video_id}.{self.file_ext}")

    def run(self):
        yt = YouTube("https://youtu.be/" + self.yt_video_id)
        media_stream = (
            yt.streams.filter(mime_type=f"{self.file_type}/mp4")
            .order_by("resolution")
            .desc()
            .first()
        )

        def show_progress_bar(stream, chunk, file_handle, bytes_remaining):
            if random.random() > 0.9:
                self.set_progress_percentage(
                    round(((stream.filesize - bytes_remaining) / stream.filesize) * 100)
                )

        if self.set_progress_percentage:
            yt.register_on_progress_callback(show_progress_bar)

        with tempfile.TemporaryDirectory() as tmpd:
            tmp_path = media_stream.download(output_path=tmpd)
            if self.set_progress_percentage:
                self.set_progress_percentage(100)
            self.output().makedirs()
            self.save_file(tmp_path)


class DownloadYoutubeVideo(YoutubeDownloader):
    file_type = "video"
    file_ext = "mp4"
    fps = luigi.FloatParameter(30.0)

    def save_file(self, file_path):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmpf:

            fps = eval(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "0",
                        "-of",
                        "csv=p=0",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=r_frame_rate",
                        file_path,
                    ]
                )
            )

            if fps != 30:
                subprocess.Popen(
                    [
                        *self.ffmpeg_bin.split(" "),
                        "-y",
                        "-i",
                        file_path,
                        "-r",
                        str(self.fps),
                        tmpf.name,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                ).communicate()
                shutil.copy(tmpf.name, self.output().path)
            else:
                shutil.copy(file_path, self.output().path)


class DownloadYoutubeAudio(YoutubeDownloader):
    fs = luigi.IntParameter(16000)
    file_type = "audio"
    file_ext = "wav"

    def save_file(self, file_path):
        with tempfile.NamedTemporaryFile(suffix="." + self.file_ext) as tmpf:
            subprocess.Popen(
                [
                    *self.ffmpeg_bin.split(" "),
                    "-y",
                    "-i",
                    file_path,
                    "-vn",
                    "-ar",
                    str(self.fs),
                    tmpf.name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ).communicate()
            shutil.copy(tmpf.name, self.output().path)


@inherits(Params)
class DownloadYoutubeCaptions(SuperTask):
    caption_type = luigi.Parameter("asr")

    def output(self):
        return self.get_output_path(self.yt_video_id + ".csv")

    def run(self):
        yt = YouTube("https://youtu.be/" + self.yt_video_id)

        for caption_url in [x.url for x in yt.captions.all() if x.code == "en"]:
            if ("kind=asr" in caption_url) is (self.caption_type == "asr"):
                page = requests.get(caption_url + "&fmt=srv2")
                sentences = []
                for w in BeautifulSoup(page.content, "lxml").findAll("text"):
                    word = w.text
                    if word.startswith("\n"):
                        continue
                    elif word.startswith("<font"):
                        word = word.split(">")[1]
                        word = word.replace("</font", "")
                    sentences.append([word, w.get("t"), w.get("d")])
                self.output().makedirs()
                with self.output().open("w") as f:
                    csv.writer(f).writerows(sentences)
