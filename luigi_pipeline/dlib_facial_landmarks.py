import glob
import os

import dlib
import luigi
from luigi.util import inherits
from luigi_pipeline import SuperTask
from luigi_pipeline.download_file import DownloadFile
from luigi_pipeline.extract_images import ExtractImages

# import cv2

DLIB_CNN_WEIGHTS = "http://arunponnusamy.com/files/mmod_human_face_detector.dat"
DLIB_SHAPE_PREDICTOR_URL = (
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
)


@inherits(ExtractImages)
class DlibFacialLandmarks(SuperTask):
    cnn_weights_url = luigi.Parameter(DLIB_CNN_WEIGHTS)
    shape_predictor_url = luigi.Parameter(DLIB_SHAPE_PREDICTOR_URL)
    resize_factor = luigi.FloatParameter(0.25)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.data_dir, "dlib_facial_landmarks", f"{self.yt_video_id}.json"
            )
        )

    def requires(self):
        return (
            self.clone(ExtractImages),
            DownloadFile(url=self.cnn_weights_url, data_dir=self.data_dir),
            DownloadFile(
                url=self.shape_predictor_url, compression="bz2", data_dir=self.data_dir
            ),
        )

    def run(self):
        image_folder, cnn_weights, shape_predictor = self.input()

        detector = dlib.cnn_face_detection_model_v1(cnn_weights.path)
        predictor = dlib.shape_predictor(shape_predictor.path)

        height = None
        width = None
        result = []

        image_files = sorted(glob.glob(os.path.join(image_folder.path, "*.jpg")))
        total_count = len(image_files)

        for i, file_path in enumerate(image_files):
            img = dlib.load_rgb_image(file_path)

            if not height or not width:
                height, width = img.shape[:2]
            resized_img = dlib.resize_image(img, self.resize_factor)

            faces = detector(img, 0)
            sub_result = []
            for f in faces:
                shape = predictor(resized_img, f.rect)
                sub_sub_result = []
                for part in shape.parts():
                    sub_sub_result += [part.x, part.y]
                sub_result.append(sub_sub_result)
            result.append(sub_result)
            self.set_status_message(f"Progressed frame: {file_path}")
            self.set_progress_percentage(round(i / total_count) * 100)

        data = {
            "video_width": width,
            "video_height": height,
            "frame_count": total_count,
            "frames": result,
        }

        self.output().makedirs()
        with self.output().open("w") as f:
            json.dump(data, f)
