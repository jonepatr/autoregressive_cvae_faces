import numpy as np

import librosa
import luigi
from luigi.util import requires
from luigi_pipeline import SuperTask
from luigi_pipeline.youtube_downloader import DownloadYoutubeAudio


@requires(DownloadYoutubeAudio)
class Autocorrelation(SuperTask):
    n_frames = luigi.FloatParameter(64)
    n_ac_coefficients = luigi.IntParameter(32)
    sampling_rate = luigi.IntParameter(16000)
    frame_duration = luigi.FloatParameter(0.016)
    hop_duration = luigi.FloatParameter(0.008)
    normalized_audio = luigi.BoolParameter(True)

    def output(self):
        return self.get_output_path(self.yt_video_id + ".npy")

    def run(self):
        y, sr = librosa.core.load(self.input().path, sr=self.sampling_rate, mono=True)

        if self.normalized_audio:
            y /= np.abs(np.max(y))

        frame_step = int(self.frame_duration * sr)
        hop_step = int(self.hop_duration * sr)

        chunked_y = librosa.util.frame(y, frame_length=frame_step, hop_length=hop_step)

        chunked_y = np.hanning(frame_step)[:, None] * chunked_y

        autocorrelation_coef = librosa.core.autocorrelate(chunked_y, axis=0)[
            : self.n_ac_coefficients
        ]

        first_element = autocorrelation_coef[:1]
        autocorr = np.divide(
            autocorrelation_coef,
            first_element,
            out=np.zeros_like(autocorrelation_coef),
            where=first_element != 0,
        )

        self.output().makedirs()
        np.save(self.output().path, autocorr.T)


@requires(DownloadYoutubeAudio)
class SpectrogramProcessing(SuperTask):
    sampling_rate = luigi.IntParameter(16000)
    n_fft = luigi.IntParameter(2048)
    n_mels = luigi.IntParameter(80)
    hop_length = luigi.IntParameter(540)
    normalized_audio = luigi.BoolParameter(True)

    def output(self):
        return self.get_output_path(self.yt_video_id + ".npy")

    def run(self):
        y, sr = librosa.core.load(self.input().path, sr=self.sampling_rate, mono=True)

        if self.normalized_audio:
            y /= np.abs(np.max(y))

        spectrogram = np.abs(
            librosa.stft(
                y=y,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=False,
                window="hann",
            )
            ** 2
        )

        melspectrogram = librosa.feature.melspectrogram(
            S=spectrogram, n_mels=self.n_mels
        )
        melspectrogram = librosa.core.power_to_db(melspectrogram)

        output = self.finish(melspectrogram)

        self.output().makedirs()
        np.save(self.output().path, output)


class MFCC(SpectrogramProcessing):
    def finish(self, melspectrogram):
        return librosa.feature.mfcc(S=melspectrogram).T


class MelSpectrogram(SpectrogramProcessing):
    def finish(self, melspectrogram):
        return melspectrogram.T
