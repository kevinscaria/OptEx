from optex.src.exemplars import BaseExemplar


class SuccessOnlyExemplars(BaseExemplar):
    # Highest reward first (B4). If tied, prefer recent.
    def select(self, buffer, instruction):
        eps = buffer.all()
        sorted_eps = sorted(
            eps,
            key=lambda e: (e.reward, e.timestamp),  # reward asc then time asc, so reverse below
            reverse=True
        )
        return sorted_eps[:min(self.k, len(sorted_eps))]
