import constants
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import tempfile
import subprocess


plt.switch_backend("agg")
plt.rcParams["animation.ffmpeg_path"] = constants.ffmpeg_bin


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
    ax.set_xlim([-1.25, 1.25])
    ax.set_ylim([-1.25, 1.25])
    ax.set_aspect("equal")


def visualize_videos(experiment, left_output, right_output, count, global_step, fps):
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
        plot_face(in_[n + 2], axes[0])
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


def finilize_visualization(
    output_name, input_name, start_time, length, audio_path=None, video_path=None
):
    video, video_filter, audio, audio_map = [], ["-map", "0:v"], [], []
    one_second = ["-ss", str(start_time), "-t", str(length)]
    if video_path:
        video_filter = [
            "-filter_complex",
            "[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]",
            "-map",
            "[vid]",
        ]
        video = [*one_second, "-i", video_path]

    if audio_path:
        audio = [*one_second, "-i", audio_path]
        audio_idx = 2 if video_path else 1
        audio_map = ["-map", f"{audio_idx}:a"]
    
    subprocess.Popen(
        [
            constants.ffmpeg_bin,
            "-y",
            "-i",
            input_name,
            *video,
            *audio,
            *video_filter,
            *audio_map,
            output_name,
        ],
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()


def plot_faces(output_file, faces, fps):
    fig, axes = plt.subplots(1, len(faces), figsize=(15 * len(faces), 10), gridspec_kw={"wspace": 0, "hspace": 0})
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    FFMpegWriter = animation.writers["ffmpeg"]
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, output_file, 100):  # tmpf.name
        for n in range(faces[0].size(0)):
            for face, ax in zip(faces, axes):
                ax.clear()
                plot_face(face[n], ax)
            writer.grab_frame()
        plt.close(fig)


def create_split_video(
    name, left_output, right_output, start_frame, video_path, audio_path, fps
):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmpf:
        plot_faces(tmpf.name, [left_output, right_output], fps)

        finilize_visualization(
            name,
            tmpf.name,
            float(start_frame) / fps,
            float(left_output.size(0)) / fps,
            audio_path=audio_path,
            video_path=video_path,
        )


def create_single_video(name, output, start_frame, audio_path, fps):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmpf:
        plot_faces(tmpf.name, [output], fps)

        finilize_visualization(
            name,
            tmpf.name,
            float(start_frame) / fps,
            float(output.size(0)) / fps,
            audio_path=audio_path,
        )
