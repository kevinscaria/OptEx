from optex.src.exemplars import BaseExemplar


class RecencyExemplars(BaseExemplar):
    def select(self, buffer, instruction):  # B3
        return buffer.recent(self.k)