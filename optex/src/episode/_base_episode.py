import numpy as np

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BaseEpisode:
    instruction: str
    plan: str
    actions: List[str]
    reward: float
    timestamp: float
    embedding: Optional[np.ndarray] = None
