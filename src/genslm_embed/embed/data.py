from dataclasses import dataclass
from typing import Iterator

import tables as tb
import torch
from torch.utils.data import BatchSampler, Dataset, SequentialSampler

from genslm_embed.utils import FilePath


@dataclass
class SequenceBatch:
    tokens: torch.Tensor
    attn_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "SequenceBatch":
        cls = self.__class__
        return cls(tokens=self.tokens.to(device), attn_mask=self.attn_mask.to(device))


class TokenizedSequenceDataset(Dataset):
    def __init__(self, file: FilePath):
        self._file = tb.File(file)
        self._attn_mask: tb.EArray = self._file.root.attn_mask
        self._tokens: tb.EArray = self._file.root.tokens

    def __getitem__(
        self, idx: int | slice | list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.from_numpy(self._tokens[idx, :])
        attn_mask = torch.from_numpy(self._attn_mask[idx, :]).bool()
        return tokens, attn_mask

    def __len__(self) -> int:
        return self._tokens.shape[0]  # type: ignore

    @staticmethod
    def collate(batch: list[torch.Tensor]) -> SequenceBatch:
        if len(batch) != 2:
            raise RuntimeError("batch must only have 2 tensors")

        return SequenceBatch(tokens=batch[0], attn_mask=batch[1])

    def __del__(self):
        self._file.close()


class SlicedBatchSampler(BatchSampler):
    def __init__(
        self, dataset: TokenizedSequenceDataset, batch_size: int, drop_last: bool
    ) -> None:
        super().__init__(SequentialSampler(dataset), batch_size, drop_last)

    def __iter__(self) -> Iterator[slice]:
        for indices in super().__iter__():
            min_idx = min(indices)
            max_idx = max(indices) + 1
            yield slice(min_idx, max_idx, 1)
