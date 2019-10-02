"""
Example template for defining a system
"""
import glob
import json
import os
from collections import OrderedDict
from itertools import combinations

import numpy as np
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning.root_module.root_module import LightningModule
from scipy.spatial import distance
from test_tube import HyperOptArgumentParser
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST

from dataset import Speech2FaceDataset
import constants
from visualize import visualize_videos


class AutoregressiveFaceVAE(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.data_files = json.load(open(self.hparams.data_files))

        self.combined_face_parts = []
        for first, last in constants.face_parts:
            self.combined_face_parts += list(combinations(range(first, last), 2))

        self.fc1 = nn.Linear(self.hparams.data_dim * 3, 70)
        self.fc31 = nn.Linear(70, self.hparams.bottleneck_size)
        self.fc32 = nn.Linear(70, self.hparams.bottleneck_size)
        self.fc4 = nn.Linear(
            self.hparams.bottleneck_size + (2 * self.hparams.data_dim), 70
        )
        self.fc6 = nn.Linear(70, self.hparams.data_dim)

        self.fc6.weight.data.fill_(0)

    def encode(self, x_t0, x_t1, x_t2):
        h1 = F.relu(self.fc1(torch.cat([x_t0, x_t1, x_t2], dim=1)))
        return self.fc31(h1), self.fc32(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_t1, x_t2):
        h3 = F.relu(self.fc4(torch.cat([z, x_t1, x_t2], dim=1)))
        return torch.tanh(self.fc6(h3))

    def forward(self, x):
        mus, logvars, zs, y_hats = [], [], [], []
        for i in range(2, x.size(1)):
            mu, logvar = self.encode(x[:, i], x[:, i - 1], x[:, i - 2])
            z = self.reparameterize(mu, logvar)
            y_hat = self.decode(z, x[:, i - 1], x[:, i - 2])
            mus.append(mu)
            logvars.append(logvar)
            zs.append(z)
            y_hats.append(y_hat)
        return (
            torch.stack(y_hats, dim=1),
            torch.stack(mus, dim=1),
            torch.stack(logvars, dim=1),
            torch.stack(zs, dim=1),
        )

    def gll_loss(self, output_x, target_x):
        GLL = 0
        for i in range(output_x[0].size(0)):
            mu_x, logvar_x = output_x[0][i], output_x[1][i]
            part1 = torch.sum(logvar_x)
            sigma = logvar_x.mul(0.5).exp_()
            part2 = torch.sum(((target_x - mu_x) / sigma) ** 2)
            GLL += 0.5 * (part1 + part2)
        return GLL

    def kld_loss(self, mu, logvar):
        return torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # , dim=1
        )

    def group_distance_loss(self, output, target):
        o_x = output[:, self.combined_face_parts][:, :, 0]
        o_y = output[:, self.combined_face_parts][:, :, 1]
        t_x = target[:, self.combined_face_parts][:, :, 0]
        t_y = target[:, self.combined_face_parts][:, :, 1]
        o_dist = torch.sqrt(torch.sum((o_x - o_y).pow(2), dim=2))
        t_dist = torch.sqrt(torch.sum((t_x - t_y).pow(2), dim=2))

        return F.mse_loss(o_dist, t_dist, reduction="sum")

    def vae_loss(self, output, target, mu, logvar):

        o = output.reshape(-1, self.hparams.frame_len-2, 70, 2)
        t = target.reshape(-1, self.hparams.frame_len-2, 70, 2)

        MSE = F.mse_loss(o, t, reduction="sum")
        # MSE = F.mse_loss(o * weights, t * weights, reduction="sum")  # / self.hparams.data_dim
        # MSE = F.mse_loss(o, t, reduction="sum") / batch_size / self.hparams.data_dim  # / self.hparams.data_dim

        # EXTRA_MSE = self.group_distance_loss(o, t) * self.hparams.group_distance_scaling

        # MSE = F.mse_loss(o, t)  # torch.mean(torch.sum((target - output) ** 2, axis=1))
        # MSE = torch.mean(F.pairwise_distance(o, t))
        # import pdb; pdb.set_trace()
        if self.current_epoch < 10:
            kl_annealing_factor = 0
        else:
            kl_annealing_factor = max(self.current_epoch - 10 / 10, 1)
        kl_annealing_factor = 1

        KLD = (
            kl_annealing_factor
            # * ((self.hparams.beta * self.hparams.bottleneck_size) / self.hparams.data_dim)
            * self.hparams.beta
            * (self.kld_loss(mu, logvar))  # / batch_size / self.hparams.bottleneck_size
        )
        # KLD = kl_annealing_factor * ((self.hparams.beta * self.hparams.bottleneck_size) / batch_size) * self.kld_loss(mu, logvar)
        # KLD = self.kld_loss(mu, logvar)
        # 0.0001 * MSE +
        # t = target.reshape(-1, 70, 2).cpu()
        # o = output.reshape(-1, 70, 2).cpu()
        # t_u = torch.cdist(t, t)
        # o_u = torch.cdist(o, o)
        # dist_loss = F.mse_loss(t_u, o_u).to(output.device)

        loss = MSE + KLD  # + EXTRA_MSE
        # loss = EXTRA_MSE

        # self.experiment.add_scalar("gll", GLL, self.global_step)
        return (
            loss.unsqueeze(0),
            MSE.unsqueeze(0),
            KLD.unsqueeze(0),
            # EXTRA_MSE.unsqueeze(0),
        )

    def training_step(self, batch, batch_nb):
        x = batch["x"]
        # cond = batch["audio_features"]
        output, mu, logvar, _ = self.forward(x)

        total_loss, mse_loss, kld_loss = self.vae_loss(
            output, x[:,2:], mu, logvar
        )
        return {
            "loss": total_loss,
            "prog": {
                "mse_loss": mse_loss,
                "kld_loss": kld_loss,
                # "extra_mse_loss": extra_mse_loss,
            },
        }

    def validation_step(self, batch, batch_nb):
        x = batch["x"]
        output, mu, logvar, z = self.forward(x)
        if batch_nb == 0:
            visualize_videos(self.experiment, x, output, 1, self.global_step, self.hparams.fps)

            # std = torch.exp(0.5 * logvar)
            # if batch_nb == 0:

            #     # Do random sampling
            #     sample = torch.randn(1, self.hparams.bottleneck_size).to(x.device)
            #     x_out = self.decode(sample)  # .numpy()

            #     # x_out = self.reparameterize(mu, logvar)
            #     sample_res = x_out.reshape(70, 2).cpu()

            #     # fig = plt.figure()
            #     fig, ax1 = plt.subplots(1, 1)
            #     # ax1.scatter(sample_res[:, 0], -sample_res[:, 1])
            #     self.plot_face(sample_res, ax1)
            #     self.experiment.add_figure("matplotlib", fig)
            #     plt.close(fig)

            for i in range(z.size(1)):
                self.experiment.add_histogram(f"z_{i}", z[:, i], self.current_epoch)

        total_loss, *_ = self.vae_loss(output, x[:,2:], mu, logvar)
        return {"val_loss": total_loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)]

    def __dataloader(self, files):

        return DataLoader(
            Speech2FaceDataset(
                files,
                data_dir=self.hparams.data_dir,
                frame_history_len=self.hparams.frame_len,
                audio_feature_type="spectrogram",
            ),
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    @ptl.data_loader
    def tng_dataloader(self):
        return self.__dataloader(self.data_files["train"])

    @ptl.data_loader
    def val_dataloader(self):
        return self.__dataloader(self.data_files["val"])

    @ptl.data_loader
    def test_dataloader(self):
        return self.__dataloader(self.data_files["test"])

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """

        parser = HyperOptArgumentParser(
            strategy=parent_parser.strategy, parents=[parent_parser]
        )

        
        parser.add_argument("--fps", default=30, type=int)
        parser.add_argument("--frame_len", default=4, type=int)
        parser.add_argument("--beta", default=1, type=float)
        parser.add_argument("--bottleneck_size", default=10, type=int)
        parser.add_argument("--group_distance_scaling", default=1, type=float)
        parser.add_argument("--audio_size", default=80, type=int)
        parser.add_argument("--data_dim", default=140, type=int)
        parser.add_argument("--data_files", default="datafiles.json", type=str)
        parser.add_argument("--data_dir", default="/data_dir", type=str)
        parser.opt_list(
            "--batch_size",
            default=256 * 4,
            type=int,
            options=[32, 64, 128, 256],
            tunable=False,
            help="batch size will be divided over all gpus being used across all nodes",
        )
        parser.opt_list(
            "--learning_rate",
            default=0.001 * 8,
            type=float,
            options=[0.0001, 0.0005, 0.001],
            tunable=True,
        )
        return parser
