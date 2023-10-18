from dataclasses import dataclass
from typing import Literal, Optional

import einops
import tables as tb
import torch
from genslm import GenSLM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from genslm_embed.utils import TABLES_COMPRESSION, FilePath, ModelTypes


@dataclass
class SequenceBatch:
    tokens: torch.Tensor
    attn_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "SequenceBatch":
        cls = self.__class__
        return cls(tokens=self.tokens.to(device), attn_mask=self.attn_mask.to(device))


class TokenizedSequenceDataset(Dataset):
    def __init__(self, file: FilePath):
        with tb.File(file) as fp:
            self.attn_mask = torch.from_numpy(fp.root.attn_mask[:])
            self.tokens = torch.from_numpy(fp.root.tokens[:])

    def __getitem__(self, idx: int | slice) -> SequenceBatch:
        return SequenceBatch(tokens=self.tokens[idx], attn_mask=self.attn_mask[idx])

    def __len__(self) -> int:
        return int(self.tokens.size(0))

    @staticmethod
    def collate(batch: list[SequenceBatch]) -> SequenceBatch:
        batch_tokens: list[torch.Tensor] = list()
        batch_attn_mask: list[torch.Tensor] = list()
        for data in batch:
            batch_tokens.append(data.tokens)
            batch_attn_mask.append(data.attn_mask)

        return SequenceBatch(
            tokens=torch.stack(batch_tokens), attn_mask=torch.stack(batch_attn_mask)
        )


class GenSLMPredictor:
    def __init__(
        self,
        data_file: FilePath,
        model_id: ModelTypes,
        model_cache_dir: FilePath,
        device: Optional[torch.device | Literal["cpu", "cuda"]] = None,
    ):
        self.dataset = TokenizedSequenceDataset(data_file)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = GenSLM(model_id, model_cache_dir).to(device=self.device)

    def dataloader(self, batch_size: int = 64) -> DataLoader[SequenceBatch]:
        return DataLoader[SequenceBatch](
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.dataset.collate,
        )

    def create_storage(self, fobj: tb.File) -> tb.EArray:
        hidden_dim: int = self.model.model.config.hidden_size  # type: ignore
        return fobj.create_earray(
            fobj.root,
            "data",
            atom=tb.Float32Atom(),
            shape=(0, hidden_dim),
            expectedrows=len(self.dataset),
            filters=TABLES_COMPRESSION,
        )

    @torch.no_grad()
    def predict_loop(self, output: FilePath, batch_size: int = 64):
        with tb.File(output, "w") as fp:
            storage = self.create_storage(fp)

            batch: SequenceBatch
            for batch in tqdm(self.dataloader(batch_size)):
                batch = batch.to(self.device)
                model_output = self.model(
                    batch.tokens, batch.attn_mask, output_hidden_states=True
                )

                emb: torch.Tensor = model_output.hidden_states[-1]

                # only average over real codons/word, not padded ones
                avg_embs: list[torch.Tensor] = list()
                for seq_idx, mask in enumerate(batch.attn_mask):
                    avg_emb = einops.reduce(
                        emb[seq_idx, mask.bool()],
                        "codons dim -> dim",
                        reduction="mean",
                    )
                    avg_embs.append(avg_emb)

                storage.append(torch.stack(avg_embs).cpu().numpy())
