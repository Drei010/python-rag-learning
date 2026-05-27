import hashlib
import math
import re
from typing import List

from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if not magnitude:
            return vector

        return [value / magnitude for value in vector]
