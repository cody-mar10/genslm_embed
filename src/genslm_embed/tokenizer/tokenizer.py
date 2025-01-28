from typing import Iterator

import numpy as np
import tables as tb
from genslm import GenSLM, SequenceDataset
from more_itertools import chunked
from pyfastatools import Parser

from genslm_embed.utils import TABLES_COMPRESSION, FilePath


def read_fasta_chunks(fasta: Parser, max_seq_size: int, n_seqs: int = 10000) -> Iterator[list[str]]:
    sequences: list[str] = list()

    for record in fasta:
        if len(record.sequence) <= max_seq_size:
            sequences.append(record.seq)
        else:
            sequences.extend("".join(chunk) for chunk in chunked(record.seq, n=max_seq_size))

        if len(sequences) >= n_seqs:
            yield sequences
            sequences = list()

    if sequences:
        yield sequences


def expected_tokens(fasta: Parser, max_seq_size: int) -> int:
    n = 0
    for record in fasta:
        if len(record.seq) <= max_seq_size:
            n += 1
        else:
            for _ in chunked(record.seq, n=max_seq_size):
                n += 1

    parser.refresh()
    return n


def tokenize(model: GenSLM, fasta: Parser, output: FilePath):
    max_seq_size = model.seq_length * 3

    rows = expected_tokens(fasta, max_seq_size)
    with tb.File(output, "w") as fp:
        token_ids = fp.create_earray(
            fp.root,
            "tokens",
            atom=tb.Int64Atom(),
            shape=(0, model.seq_length),
            expectedrows=rows,
            filters=TABLES_COMPRESSION,
        )

        attn = fp.create_earray(
            fp.root,
            "attn_mask",
            atom=tb.Int64Atom(),
            shape=(0, model.seq_length),
            expectedrows=rows,
            filters=TABLES_COMPRESSION,
        )

        for sequences in read_fasta_chunks(fasta, max_seq_size=max_seq_size):
            dataset = SequenceDataset(
                sequences=sequences,
                seq_length=model.seq_length,
                tokenizer=model.tokenizer,
                verbose=False,
            )

            tokens = np.stack(
                [batch["input_ids"].squeeze() for batch in dataset.batch_encodings]  # type: ignore
            )
            mask = np.stack(
                [batch["attention_mask"].squeeze() for batch in dataset.batch_encodings]  # type: ignore
            )

            token_ids.append(tokens)
            attn.append(mask)
