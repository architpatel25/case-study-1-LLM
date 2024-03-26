from base import BASE, abstractmethod
from txtai import Embeddings
from txtai.pipeline import Extractor
from typing import List
from pydantic import BaseModel

class Data(BaseModel):
    id: int
    text: str

class IDataService(BASE):
    @abstractmethod
    def index_data(self, data: List[Data]):
        pass

    @abstractmethod
    def search(self, query: str):
        pass

    @abstractmethod
    def update_data(self, idx: int, text: str):
        pass

    @abstractmethod
    def delete_data(self, idx: int):
        pass

    @abstractmethod
    def rag_generate(self, question: str):
        pass

class TxtaiDataService(IDataService):
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.extractor = Extractor(self.embeddings, "google/flan-t5-large")

    def index_data(self, data: List[Data]):
        texts = [d.text for d in data]
        self.embeddings.index(texts)

    def search(self, query: str):
        result = self.embeddings.search(query, 1)
        return result

    def update_data(self, idx: int, text: str):
        self.embeddings.upsert([(idx, text, None)])

    def delete_data(self, idx: int):
        self.embeddings.delete([idx])

    def rag_generate(self, question: str):
        context = lambda q: [{"query": q, "question": f"Answer the following question using the context below.\nQuestion: {q}\nContext:"}]
        return self.extractor(context(question))[0]
