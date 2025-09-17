import random
from optex.src.exemplars import BaseExemplar


class RandomExemplars(BaseExemplar):
    def select(self, buffer, instruction):  # B1
        eps = buffer.all()
        if len(eps) <= self.k: return eps
        return random.sample(eps, self.k)
