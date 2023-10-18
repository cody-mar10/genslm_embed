import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastatools import FastaFile
from genslm import GenSLM

from genslm_embed.tokenizer.tokenizer import tokenize
from genslm_embed.utils import FilePath, ModelArgs, add_model_args


@dataclass
class Args(ModelArgs):
    fasta: FilePath
    output: FilePath


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-f",
        "--fasta",
        type=Path,
        metavar="FILE",
        required=True,
        help="input FASTA file of genome sequences to embed",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        default=Path("tokens.h5"),
        help="output file for tokens (default: %(default)s)",
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

    model = GenSLM(model_id=args.model_id, model_cache_dir=args.model_cache)
    fasta = FastaFile(args.fasta)

    tokenize(model, fasta, args.output)


if __name__ == "__main__":
    main()
