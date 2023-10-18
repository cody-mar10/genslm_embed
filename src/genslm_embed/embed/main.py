import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, get_args

import tables as tb

from genslm_embed.embed.embed import GenSLMPredictor
from genslm_embed.utils import Devices, FilePath, ModelArgs, add_model_args


@dataclass
class Args(ModelArgs):
    token_file: FilePath
    batch_size: int
    output: FilePath
    device: Optional[Devices]

    def __post_init__(self):
        super().__post_init__()

        if not Path(self.token_file).exists():
            raise FileNotFoundError(f"{self.token_file} does not exist")

        with tb.File(self.token_file) as fp:
            for attr in ("tokens", "attn_mask"):
                if not hasattr(fp.root, attr):
                    raise RuntimeError(
                        f"{self.token_file} does not have the `{attr}` dataset"
                    )


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-t",
        "--token-file",
        type=Path,
        metavar="FILE",
        required=True,
        help="input tokenized dataset file in .h5 format. Must have `tokens` and `attn_mask` datasets",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        metavar="INT",
        default=64,
        help="inference batch size in number of sequences (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        default=Path("predictions.h5"),
        help="output file for predictions (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--device",
        metavar="DEVICE",
        default=None,
        choices=get_args(Devices) + (None,),
        help="accelerator device to use (default: %(default)s -- auto detect device)",
    )

    add_model_args(parser)


def parse_args(parser: Optional[argparse.ArgumentParser] = None) -> Args:
    if parser is None:
        parser = argparse.ArgumentParser()

    add_args(parser)
    args = parser.parse_args()
    return Args.from_args(args)


def main(args: Optional[Args] = None):
    if args is None:
        args = parse_args()

    predictor = GenSLMPredictor(
        data_file=args.token_file,
        model_id=args.model_id,
        model_cache_dir=args.model_cache,
        device=args.device,
    )

    predictor.predict_loop(output=args.output, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
