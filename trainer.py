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

    exp = Experiment(
        name=hparams.tt_name,
        debug=hparams.debug,
        save_dir=hparams.tt_save_path,
        version=hparams.hpc_exp_number,
        autosave=False,
        description=hparams.tt_description,
    )

    exp.argparse(hparams)
    exp.save()

    model = FaceVAE(hparams)

    early_stop = EarlyStopping(
        monitor="avg_val_loss", patience=3, verbose=True, mode="min"
    )

    model_save_path = "{}/{}/{}".format(hparams.model_save_path, exp.name, exp.version)
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        verbose=True,
        monitor="avg_val_loss",
        mode="min",
    )

    trainer = Trainer(
        experiment=exp,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        gpus=hparams.gpus,
        distributed_backend=hparams.dist_backend,
        # val_check_interval=0.5,
        # distributed_backend="dp",
        # overfit_pct=0.01
    )

    trainer.fit(model)


if __name__ == "__main__":

    SEED = 2538
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules["__main__"].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy="random_search", add_help=False)
    add_default_args(parent_parser, root_dir, rand_seed=SEED)
    parent_parser.add_argument(
        "--dist_backend",
        type=str,
        default="dp",
        help="When using multiple GPUs set Trainer(distributed_backend=dp) (or ddp)",
    )
    # allow model to overwrite or extend args
    parser = FaceVAE.add_model_specific_args(parent_parser, root_dir)

    hyperparams = parser.parse_args()
    print(hyperparams)
    # train model
    main(hyperparams)

