from typing import List

from optex.src.episode import BaseEpisode


class BaseExemplarBuffer:
    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self.episodes: List[BaseEpisode] = []

    def add(self, ep: BaseEpisode):
        self.episodes.append(ep)
        if len(self.episodes) > self.max_size:
            # simple eviction: drop oldest
            self.episodes.pop(0)

    def all(self) -> List[BaseEpisode]:
        return self.episodes

    def recent(self, k: int) -> List[BaseEpisode]:
        return list(sorted(self.episodes, key=lambda e: e.timestamp, reverse=True))[:k]