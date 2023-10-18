import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, get_args

import tables

ModelTypes = Literal[
    "genslm_25M_patric", "genslm_250M_patric", "genslm_2.5B_patric", "genslm_25B_patric"
]

FilePath = str | Path
Devices = Literal["cpu", "cuda"]

TABLES_COMPRESSION = tables.Filters(complevel=4, complib="blosc:lz4")


@dataclass
class ModelArgs:
    model_id: ModelTypes
    model_cache: FilePath

    def __post_init__(self):
        if self.model_id not in get_args(ModelTypes):
            raise ValueError(
                f"{self.model_id} is not one of the following genslm models: {ModelTypes}"
            )

        if not Path(self.model_cache).exists():
            raise RuntimeError(f"{self.model_cache} does not exist")

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        kwargs = {field.name: getattr(args, field.name) for field in fields(cls)}
        return cls(**kwargs)


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-m",
        "--model-cache",
        type=Path,
        metavar="DIR",
        default=Path("."),
        help="genslm model dir with pre-trained model weights (default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--model-id",
        metavar="MODEL",
        choices=get_args(ModelTypes),
        default="genslm_25M_patric",
        help="genslm model (default: %(default)s) [choices: %(choices)s]",
    )
