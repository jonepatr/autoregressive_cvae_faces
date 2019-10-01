import os
import sys
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.arg_parse import add_default_args
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from test_tube import Experiment, HyperOptArgumentParser

from models.vae_faces import FaceVAE



def main(hparams):
    parser = HyperOptArgumentParser(strategy="random_search")

    parser.opt_list(
        "--learning_rate",
        default=0.001 * 8,
        type=float,
        options=[0.0001, 0.0005, 0.001],
        tunable=True,
    )

    parser.add_argument("--gpus", type=str, default="0,1,2,3")

    root_dir = os.path.dirname(os.path.realpath(__file__))
    demo_log_dir = os.path.join(root_dir, "results")
    checkpoint_dir = os.path.join(demo_log_dir, "model_weights")
    test_tube_dir = os.path.join(demo_log_dir, "test_tube_data")

    parser.add_argument(
        "--test_tube_save_path",
        type=str,
        default=test_tube_dir,
        help="where to save logs",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=checkpoint_dir,
        help="where to save model",
    )
    parser.add_argument("--experiment_name", type=str, default="vae")
    hparams = parser.parse_args()

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    print("loading model...")
    model = FaceVAE(hparams)
    print("model built")

    # ------------------------
    # 2 INIT TEST TUBE EXP
    # ------------------------

    # init experiment
    exp = Experiment(
        # version=4,
        name=hparams.experiment_name,
        save_dir=hparams.test_tube_save_path,
        autosave=False,
        description="test demo",
    )

    exp.argparse(hparams)
    exp.save()

    # ------------------------
    # 3 DEFINE CALLBACKS
    # ------------------------
    model_save_path = "{}/{}/{}".format(hparams.model_save_path, exp.name, exp.version)
    early_stop = EarlyStopping(
        monitor="avg_val_loss", patience=3, verbose=True, mode="max"
    )

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        # verbose=True,
        monitor="avg_val_loss",
        mode="min",
    )

    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        experiment=exp,
        checkpoint_callback=checkpoint,
        # early_stop_callback=early_stop,
        gpus=hparams.gpus,
        val_check_interval=0.5,
        distributed_backend="dp",
        # overfit_pct=0.01
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':

    SEED = 2538
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=False)
    add_default_args(parent_parser, root_dir, rand_seed=SEED)

    exit()
    # allow model to overwrite or extend args
    parser = FaceVAE.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    # print(hyperparams)
    exit()
    # train model
    main(hyperparams)

