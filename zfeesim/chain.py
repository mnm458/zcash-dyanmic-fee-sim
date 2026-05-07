"""Chain state: ordered list of blocks."""

from __future__ import annotations

from .types import Block


class Chain:
    def __init__(self) -> None:
        self._blocks: list[Block] = []

    def append(self, block: Block) -> None:
        self._blocks.append(block)

    @property
    def height(self) -> int:
        return len(self._blocks)

    @property
    def blocks(self) -> list[Block]:
        return self._blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, idx):
        return self._blocks[idx]
