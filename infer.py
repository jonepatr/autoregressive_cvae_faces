from models.vae_faces_autoregressive import AutoregressiveFaceVAE
from models.vae_faces import FaceVAE
import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider
import ipywidgets as widgets
import visualize

experiment = 31

file_ = glob.glob(f"/workspace/model_weights/pt_test/{experiment}/_ckpt_epoch_*.ckpt")[
    0
]

pretrained_model = AutoregressiveFaceVAE.load_from_metrics(
    weights_path=file_,
    tags_csv=f"/workspace/test_tube_logs/pt_test/version_{experiment}/meta_tags.csv",
    on_gpu=False
    #     map_location=None
)

# predict
pretrained_model.eval()
pretrained_model.freeze()
data = pretrained_model.val_dataloader

dp = data.dataset[0]
dp2 = torch.tensor(np.expand_dims(dp["x"][:3], 0))


results = pretrained_model.infer(dp2, 100)

pretrained_model.video_writer.create_video(f"videos/testing.mp4", results, results, 0, 0, 0)

