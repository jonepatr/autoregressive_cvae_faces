import constants
import torch
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("agg")


def plot_face(x, ax):
    # draw lines
    for first, last in (
        constants.jaw,
        constants.left_eyebrow,
        constants.right_eyebrow,
        constants.vertical_nose,
        constants.horizontal_nose,
    ):
        ax.plot(x[first:last, 0], -x[first:last, 1])

    # full circles
    for first, last in (
        constants.left_eye,
        constants.right_eye,
        constants.outer_mouth,
        constants.inner_mouth,
    ):
        points = x[list(range(first, last)) + [first]]
        ax.plot(points[:, 0], -points[:, 1])

    ax.scatter(x[68:70, 0], -x[68:70, 1])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect("equal")


def visualize_videos(experiment, left_output, right_output, count, global_step, fps
):
    rand_ints = torch.randint(left_output.size(0), (count,))
    fig2, axes = plt.subplots(
        count, 2, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0}
    )
    imgs = []
    i = 0
    # for i, rand_int in enumerate(rand_ints):
    in_ = left_output[rand_ints[0]].reshape(-1, 70, 2).detach().cpu()
    out = right_output[rand_ints[0]].reshape(-1, 70, 2).detach().cpu()
    for n in range(out.size(0)):
        fig2.clf()
        plot_face(in_[n+2], axes[0])
        plot_face(out[n], axes[1])
        fig2.canvas.draw()
        imgs.append(np.array(fig2.canvas.renderer.buffer_rgba()))
    
    # Create a B(atch) x T(ime) x C(hannel) x H(eight) x W(idth) array
    video = np.moveaxis(np.expand_dims(np.stack(imgs), 1), 4, 1)
    
    experiment.add_video("encoder/decoder", video, global_step)
    plt.close(fig2)


def visualize_pairs(experiment, left_output, right_output, count, global_step):
    rand_ints = torch.randint(left_output.size(0), (count,))
    fig2, axes = plt.subplots(
        count, 2, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0}
    )
    for i, rand_int in enumerate(rand_ints):
        in_ = left_output[rand_int].reshape(70, 2).detach().cpu()
        out = right_output[rand_int].reshape(70, 2).detach().cpu()
        plot_face(in_, axes[i][0])
        plot_face(out, axes[i][1])

    experiment.add_figure("encoder/decoder", fig2, global_step)
    plt.close(fig2)
