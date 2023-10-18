import argparse

import genslm_embed.embed as embed
import genslm_embed.tokenizer as tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    tokenizer_parser = subparsers.add_parser("tokenizer")
    embed_parser = subparsers.add_parser("embed")

    tokenizer.main.add_args(tokenizer_parser)
    embed.main.add_args(embed_parser)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "tokenizer":
        config = tokenizer.main.Args.from_args(args)
        tokenizer.main.main(config)
    elif args.command == "embed":
        config = embed.main.Args.from_args(args)
        embed.main.main(config)


if __name__ == "__main__":
    main()
