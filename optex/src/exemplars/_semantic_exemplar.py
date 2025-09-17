import numpy as np

from optex.src.exemplars import BaseExemplar
from optex.src.exemplars import RandomExemplars


class SemanticExemplars(BaseExemplar):
    # Nearest by instruction embedding cosine similarity (B2)
    def select(self, buffer, instruction):
        eps = buffer.all()
        if not eps or self.k == 0:
            return []

        if self.embedder is None:
            return RandomExemplars(self.k).select(buffer, instruction)

        q = self.embedder.encode([instruction])[0]  # (d,)
        # Ensure episode embeddings exist
        embs, items = [], []
        for e in eps:
            if e.embedding is None:
                e.embedding = self.embedder.encode([e.instruction])[0]
            embs.append(e.embedding);
            items.append(e)
        embs = np.stack(embs, axis=0)  # (N, d)
        qn = q / (np.linalg.norm(q) + 1e-9)
        En = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        sims = (En @ qn)
        top_idx = np.argsort(-sims)[:min(self.k, len(items))]
        return [items[i] for i in top_idx]
    