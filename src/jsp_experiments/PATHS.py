import os
import pathlib as pl

PROJECT_DIR = pl.Path(os.path.abspath(__file__)) \
        .parent \
        .parent \
        .parent \

WAND_OUTPUT_PATH = PROJECT_DIR.joinpath("out").joinpath("wandb")