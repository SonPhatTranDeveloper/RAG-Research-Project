"""
Author: Son Phat Tran
This file contains various implementation of document retriever
"""
from abc import ABC, abstractmethod
from typing import List

from langchain.schema import Document
from langchain_core.embeddings.embeddings import Embeddings

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma, VectorStore

from text_segmentor import TextChunk


class DocumentRetrieverBuilder(ABC):
    @abstractmethod
    def build(self, chunks: List[TextChunk]):
        pass


class BM25RetrieverBuilder(DocumentRetrieverBuilder):
    def __init__(self, k: int):
        """
        Create BM25 retriever builder
        :param k: the nearest neighbor parameters
        """
        self.k = k

    def build(self, chunks: List[TextChunk]):
        """
                Build BM25 Text Retriever from Text Chunks
                :param k: nearest neighbor parameter
                :param chunks
                :return: BM25 Retriever
                """
        # Create a list of document
        documents = [Document(page_content=chunk.content,
                              metadata={
                                  "source": chunk.document_name,
                                  "location": chunk.chunk_location
                              })
                     for chunk in chunks]

        # Initialize the retriever with Document objects
        return BM25Retriever.from_documents(documents, k=self.k)


class ChromaRetrieverBuilder(DocumentRetrieverBuilder):
    def __init__(self, embedding: Embeddings, k: int):
        """
        Create BM25 retriever builder
        :param embedding: embedding function
        :param k: the nearest neighbor parameters
        """
        self.embedding = embedding
        self.k = k

    def build(self, chunks: List[TextChunk]):
        """
        Build Chrome-based Text Retriever from Text Chunks
        :param k: nearest neighbor parameters
        :param embedding: embedding for the vector store
        :param chunks
        :return: Chrome Vector Store Retriever
        """
        # Create a list of document
        documents = [Document(page_content=chunk.content,
                              metadata={
                                  "source": chunk.document_name,
                                  "location": chunk.chunk_location
                              })
                     for chunk in chunks]

        # Create Chroma retriever from document
        chroma_vector_store = Chroma.from_documents(documents, self.embedding)
        return chroma_vector_store.as_retriever(search_kwargs={"k": self.k})


if __name__ == "__main__":
    # Read markdown file
    document_source = "data/markdowns/text/Chapter_3.md"
    with open(document_source, "r") as input_file:
        text = input_file.read()

    # Chunk the text
    from text_segmentor import MarkDownHeaderSegmentor

    segmentor = MarkDownHeaderSegmentor()
    chunks = segmentor.segment(text)
    for chunk in chunks:
        chunk.document_name = document_source

    # Import embedding
    from llm import LargeLanguageModelBuilder

    embedding, llm = LargeLanguageModelBuilder.get_google_gemini_model("AIzaSyDTzDiLUvpSxZfEeEh91uQ8HHadE1VRJzg")

    # Create retriever
    retriever = ChromaRetrieverBuilder(embedding, k=1).build(chunks)
    print(retriever)
