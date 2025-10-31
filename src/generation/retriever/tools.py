"""Retrieval tools for F1 regulations (RAG).

Provides a document_retriever_F1 factory that returns a retriever object
compatible with LangChain workflows. This preserves the logic originally
implemented in query_f1_regulations.py: PGVector store, base retriever,
CrossEncoder reranker and ContextualCompressionRetriever.
"""
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class DocumentRetrieverF1:
    """Encapsulate the retrieval pipeline for F1 regulations."""

    def __init__(
        self,
        connection: str = "postgresql+psycopg://postgres:example@localhost:54320/mydb",
        collection_name: str = "F1_Regulations",
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "mps",
    ):
        self.connection = connection
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.device = device

        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": self.device},
        )

        # Initialize PGVector vectorstore
        self.vectorstore = PGVector(
            embeddings=self.embedding_model,
            collection_name=self.collection_name,
            connection=self.connection,
            use_jsonb=True,
        )

        # Base retriever (retrieve more candidates for reranking. 5 by default)
        self.base_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        # Cross-encoder reranker and compressor
        self.cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.compressor = CrossEncoderReranker(model=self.cross_encoder, top_n=3)

        # Contextual compression retriever combining retriever + reranker
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.base_retriever,
        )

    def get_relevant_documents(self, query: str):
        """Return relevant documents for a given query.

        Mirrors LangChain retriever API-ish semantics.
        """
        
        return self.retriever.invoke(query)


def document_retriever_F1(connection: Optional[str] = None, collection_name: Optional[str] = None, **kwargs) -> DocumentRetrieverF1:
    """Factory that returns a configured DocumentRetrieverF1 instance.

    Args:
        connection: SQLAlchemy connection string for Postgres.
        collection_name: Collection/table name in PGVector.
        kwargs: Passed to DocumentRetrieverF1 (embedding model, device, ...)
    """
    conn = connection or "postgresql+psycopg://postgres:example@localhost:54320/mydb"
    coll = collection_name or "F1_Regulations"
    return DocumentRetrieverF1(connection=conn, collection_name=coll, **kwargs)
