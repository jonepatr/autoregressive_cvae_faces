import glob
from collections import defaultdict
from os.path import basename

import numpy as np
from torch.utils.data import Dataset

from luigi_pipeline.audio_processing import MelSpectrogram
from luigi_pipeline.post_process_openpose import PostProcessOpenpose
from luigi_pipeline.youtube_downloader import DownloadYoutubeAudio, DownloadYoutubeVideo
from tqdm import tqdm


class Speech2FaceDataset(Dataset):
    def __init__(
        self, file_names, data_dir=None, frame_history_len=None, audio_feature_type="spectrogram"
    ):

        self.data = []

        self.face_data = []
        self.audio_features_data = []

        for _, filepath in enumerate(tqdm(file_names)):
            openpose_file_path = PostProcessOpenpose(
                    data_dir=data_dir,
                    yt_video_id=filepath
            ).output().path
            if audio_feature_type == "spectrogram":
                ms = MelSpectrogram(
                    data_dir=data_dir,
                    yt_video_id=filepath,
                    hop_length=540,
                    sampling_rate=16200,
                    n_fft=540,
                )
                audio_feature_data = np.load(ms.output().path).astype(np.float32)
            audio_path = (
                DownloadYoutubeAudio(data_dir=data_dir, yt_video_id=filepath)
                .output()
                .path
            )
            video_path = (
                DownloadYoutubeVideo(data_dir=data_dir, yt_video_id=filepath)
                .output()
                .path
            )

            self.audio_features_data.append(audio_feature_data)
            self.face_data.append(defaultdict(list))

            openface_data = np.load(openpose_file_path, allow_pickle=True)
            for frame, all_faces in openface_data.item().items():
                all_faces_len = len(all_faces)
                face_d = []
                if all_faces_len > frame_history_len:
                    for i in range(all_faces_len):
                        first_frame = frame + i
                        # append is a hack so that instead of 7 we have 8 values
                        face_d.append(all_faces[i].reshape(140).astype("float32"))

                        if first_frame + frame_history_len < all_faces_len:

                            face = (
                                len(self.face_data) - 1,
                                frame,
                                i,
                                i + frame_history_len,
                            )
                            audio_features = (
                                len(self.audio_features_data) - 1,
                                first_frame,
                                first_frame + frame_history_len,
                            )

                            if (
                                all_faces_len > i + frame_history_len
                                and len(self.audio_features_data[-1])
                                > first_frame + frame_history_len
                            ):
                                self.data.append(
                                    (
                                        face,
                                        audio_features,
                                        first_frame,
                                        audio_path,
                                        video_path,
                                    )
                                )
                    self.face_data[-1][frame] = np.array(face_d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        face, audio_features, first_frame, audio_path, video_path = self.data[index]

        face_index, frame, face_start, face_stop = face
        audio_feature_index, audio_feature_start, audio_feature_stop = audio_features

        return {
            "x": self.face_data[face_index][frame][
                face_start:face_stop
            ],  # .transpose(1, 0, 2)
            "audio_features": self.audio_features_data[audio_feature_index][
                audio_feature_start:audio_feature_stop
            ],
            "first_frame": first_frame,
            "audio_path": audio_path,
            "video_path": video_path,
            "y": 1,
        }

