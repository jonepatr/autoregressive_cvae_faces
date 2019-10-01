"""
Example template for defining a system
"""
import glob
import json
import os
from collections import OrderedDict
from itertools import combinations

import matplotlib.pyplot as plt
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

plt.switch_backend("agg")





class FaceVAE(LightningModule):
    """
    Sample model to show how to define a template
    """

    jaw = (0, 17)
    left_eyebrow = (17, 22)
    right_eyebrow = (22, 27)
    vertical_nose = (27, 31)
    horizontal_nose = (31, 36)
    left_eye = (36, 42)
    right_eye = (42, 48)
    outer_mouth = (48, 60)
    inner_mouth = (60, 68)

    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super().__init__()
        self.hparams = hparams
        # self.audio_size = hparams.audio_size

        self.data_files = json.load(open(self.hparams.data_files))

        # if you specify an example input, the summary will show input/output for each layer
        # self.example_input_array = torch.rand(80, 140)

        face_parts = (
            self.left_eye,
            self.right_eye,
            self.outer_mouth,
            self.inner_mouth,
            self.jaw,
            self.left_eyebrow,
            self.right_eyebrow,
            self.vertical_nose,
            self.horizontal_nose,
        )
        self.combined_face_parts = []
        for first, last in face_parts:
            self.combined_face_parts += list(combinations(range(first, last), 2))

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """

        self.fc1 = nn.Linear(self.hparams.data_dim, 70)
        # self.fc2 = nn.Linear(100, 80)
        # self.fc22 = nn.Linear(80, 40)
        self.fc31 = nn.Linear(70, self.hparams.bottleneck_size)
        self.fc32 = nn.Linear(70, self.hparams.bottleneck_size)
        self.fc4 = nn.Linear(self.hparams.bottleneck_size, 70)
        # self.fc4 = nn.Linear(40, 80)
        # self.fc5 = nn.Linear(80, 100)
        # self.fc61 = nn.Linear(40, 140)
        self.fc6 = nn.Linear(70, self.hparams.data_dim)

        # self.fc1 = nn.Linear(self.hparams.data_dim, 1000)
        # self.fc2 = nn.Linear(1000, 1000)
        # self.fc22 = nn.Linear(1000, 1000)
        # self.fc31 = nn.Linear(1000, self.hparams.bottleneck_size)
        # self.fc32 = nn.Linear(1000, self.hparams.bottleneck_size)
        # self.fc40 = nn.Linear(self.hparams.bottleneck_size, 1000)
        # self.fc4 = nn.Linear(1000, 1000)
        # self.fc5 = nn.Linear(1000, 1000)
        # self.fc6 = nn.Linear(1000, self.hparams.data_dim)
        self.fc6.weight.data.fill_(0)

    # ---------------------
    # TRAINING
    # ---------------------

    # def filmit(self, f):
    #     return torch.chunk(f, 2, dim=1)

    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                self.experiment.add_histogram(
                    tag=name, values=grads, global_step=self.trainer.global_step
                )

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        # h2 = F.relu(self.fc2(h1))
        # h3 = F.relu(self.fc22(h2))
        return self.fc31(h1), self.fc32(h1)

    # def encode(self, x, cond):
    #     h1 = F.relu(self.fc1(self.film(x, *self.filmit(self.fc1_film(cond)))))
    #     h2 = F.relu(self.fc2(self.film(h1, *self.filmit(self.fc2_film(cond)))))
    #     return self.fc31(torch.sigmoid(self.film(h2, *self.filmit(self.fc31_film(cond))))), F.relu(self.fc32(torch.sigmoid(self.film(h2, *self.filmit(self.fc32_film(cond))))))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        # h4 = F.relu(self.fc4(h3))
        # h5 = F.relu(self.fc5(h4))
        return torch.tanh(self.fc6(h3))  # , self.fc62(h5)

    # def decode(self, z, cond):
    #     h3 = F.relu(self.fc4(self.film(z, *self.filmit(self.fc4_film(cond)))))
    #     h4 = F.relu(self.fc5(self.film(h3, *self.filmit(self.fc5_film(cond)))))
    #     return torch.tanh(self.fc6(self.film(h4, *self.filmit(self.fc6_film(cond)))))

    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y_hat = self.decode(z)
        # y_hat = self.decode(mu)

        return y_hat, mu, logvar, z

    def gll_loss(self, output_x, target_x):
        GLL = 0
        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        o_x = output[:, self.combined_face_parts][:,:,0]
        o_y = output[:, self.combined_face_parts][:,:,1]
        t_x = target[:, self.combined_face_parts][:,:,0]
        t_y = target[:, self.combined_face_parts][:,:,1]
        # import pdb; pdb.set_trace()
        o_dist = torch.sqrt(torch.sum((o_x - o_y).pow(2), dim=2))
        t_dist = torch.sqrt(torch.sum((t_x - t_y).pow(2), dim=2))

        return F.mse_loss(o_dist, t_dist, reduction="sum")

    def vae_loss(self, output, target, mu, logvar):
        # MSE = F.mse_loss(output, target)
        # batch_size = output.size(0)

        # GLL = self.gll_loss(output, target) / batch_size

        # output.reshape(-1, 70, 2)[:, list(range(48, 60)) + list(range(60, 68))] *= 100
        # target.reshape(-1, 70, 2)[:, list(range(48, 60)) + list(range(60, 68))] *= 100
        # target.reshape(-1, 70, 2)[:, list(range(48, 60))] *= 100
        # import pdb; pdb.set_trace()
        o = output.reshape(-1, 70, 2)
        t = target.reshape(-1, 70, 2)

        # weights = torch.ones_like(o)
        # weights[:, list(range(48, 60)) + list(range(60, 68))] = 100

        # import pdb; pdb.set_trace()
        MSE = F.mse_loss(o, t, reduction="sum")
        # MSE = F.mse_loss(o * weights, t * weights, reduction="sum")  # / self.hparams.data_dim
        # MSE = F.mse_loss(o, t, reduction="sum") / batch_size / self.hparams.data_dim  # / self.hparams.data_dim

        EXTRA_MSE = self.group_distance_loss(o, t) * self.hparams.group_distance_scaling

        # MSE = F.mse_loss(o, t)  # torch.mean(torch.sum((target - output) ** 2, axis=1))
        # MSE = torch.mean(F.pairwise_distance(o, t))
        # import pdb; pdb.set_trace()
        if self.current_epoch < 10:
            kl_annealing_factor = 0
        else:
            kl_annealing_factor = max(self.current_epoch - 10 / 10, 1)
        kl_annealing_factor = 1
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
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

        loss = MSE + KLD + EXTRA_MSE
        # loss = EXTRA_MSE

        # self.experiment.add_scalar("gll", GLL, self.global_step)
        return (
            loss.unsqueeze(0),
            MSE.unsqueeze(0),
            KLD.unsqueeze(0),
            EXTRA_MSE.unsqueeze(0),
        )

    def visualize_pairs(self, left_output, right_output, count):
        rand_ints = torch.randint(left_output.size(0), (count,))
        fig2, axes = plt.subplots(
            count, 2, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0}
        )
        for i, rand_int in enumerate(rand_ints):
            in_ = left_output[rand_int].reshape(70, 2).detach().cpu()
            out = right_output[rand_int].reshape(70, 2).detach().cpu()
            self.plot_face(in_, axes[i][0])
            self.plot_face(out, axes[i][1])

        self.experiment.add_figure("encoder/decoder", fig2, self.global_step)
        plt.close(fig2)

    def training_step(self, batch, batch_nb):
        x = batch["x"].squeeze(1)
        output, mu, logvar, _ = self.forward(x)

        total_loss, mse_loss, kld_loss, extra_mse_loss = self.vae_loss(
            output, x, mu, logvar
        )
        return {
            "loss": total_loss,
            "prog": {
                "mse_loss": mse_loss,
                "kld_loss": kld_loss,
                "extra_mse_loss": extra_mse_loss,
            },
        }

    def plot_face(self, x, ax):
        # draw lines
        for first, last in (
            self.jaw,
            self.left_eyebrow,
            self.right_eyebrow,
            self.vertical_nose,
            self.horizontal_nose,
        ):
            ax.plot(x[first:last, 0], -x[first:last, 1])

        # full circles
        for first, last in (
            self.left_eye,
            self.right_eye,
            self.outer_mouth,
            self.inner_mouth,
        ):
            points = x[list(range(first, last)) + [first]]
            ax.plot(points[:, 0], -points[:, 1])

        ax.scatter(x[68:70, 0], -x[68:70, 1])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect("equal")

    def validation_step(self, batch, batch_nb):
        x = batch["x"].squeeze(1)
        output, mu, logvar, z = self.forward(x)
        if batch_nb == 0:
            self.visualize_pairs(x, output, 5)

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

        total_loss, *_ = self.vae_loss(output, x, mu, logvar)
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
                frame_history_len=1,
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

        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

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
        return parser
