from typing import List

from optex.src.episode import BaseEpisode
from optex.src.buffer import BaseExemplarBuffer


class BaseExemplar:
    def __init__(self, k: int, embedder=None):
        self.k = k
        self.embedder = embedder

    def select(self, buffer: BaseExemplarBuffer, instruction: str) -> List[BaseEpisode]:
        raise NotImplementedError
