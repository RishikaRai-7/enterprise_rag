from typing import List

import chromadb

from app.config.settings import CHROMA_DB_PATH


class ChromaVectorStore:
    """
    Wrapper around ChromaDB.

    Responsible only for storing
    and retrieving embeddings.
    """

    def __init__(
        self,
        collection_name: str = "enterprise_rag"
    ):

        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(

        self,

        ids: List[str],

        documents: List[str],

        embeddings: List[List[float]],

        metadatas: List[dict],

    ):

        self.collection.add(

            ids=ids,

            documents=documents,

            embeddings=embeddings,

            metadatas=metadatas,

        )

    def query(

        self,

        embedding: List[float],

        top_k: int = 10,

    ):

        return self.collection.query(

            query_embeddings=[embedding],

            n_results=top_k,

        )

    def count(self):

        return self.collection.count()

    def delete(self):

        self.client.delete_collection(
            self.collection.name
        )