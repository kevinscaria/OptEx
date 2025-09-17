from optex.src.exemplars import BaseExemplar


class NoExemplars(BaseExemplar):
    def select(self, buffer, instruction):  # B0
        return []
