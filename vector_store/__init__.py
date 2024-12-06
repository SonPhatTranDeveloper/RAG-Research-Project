"""
Author: Son Phat Tran
This file implements the various vector store for storing the text chunks
"""
import json
from abc import ABC, abstractmethod
from typing import List

from langchain_chroma import Chroma

from text_segmentor import TextChunk


class VectorStore(ABC):
    @abstractmethod
    def load(self) -> List[TextChunk]:
        pass

    @abstractmethod
    def save(self, chunks: List[TextChunk]) -> None:
        pass


class JsonVectorStore(VectorStore, ABC):
    def __init__(self, vector_store_path: str):
        """
        Create a local JSON-based vector store
        :param vector_store_path: path of the local file
        """
        self.file_path = vector_store_path

    def save(self, chunks: List[TextChunk]) -> None:
        """
        Save chunks to a JSON file
        :param chunks: chunks to save
        :return: None
        """
        # Create a list containing the chunks as dictionaries
        chunk_array = [
            {
                "text": chunk.content,
                "source": chunk.document_name,
                "location": chunk.chunk_location
            }
            for index, chunk in enumerate(chunks)
        ]

        # Save the array
        with open(self.file_path, "w") as output_file:
            json.dump(chunk_array, output_file)

    def load(self) -> List[TextChunk]:
        """
        Load chunks from a JSON file
        :return: A list of TextChunk loaded from a JSON file
        """
        # Load the JSON array
        with open(self.file_path, "r") as f:
            chunk_array = json.load(f)

        # Convert to a list of chunk
        return [
            TextChunk(chunk_dict["text"], chunk_dict["source"], chunk_dict["location"])
            for chunk_dict in chunk_array
        ]


class ChromaVectorStore(VectorStore, ABC):
    def __init__(self, vector_store_directory_path: str, embedding_function):
        """
        Create a Chroma-based vector store
        :param vector_store_directory_path: path of the local file (this should be
        the directory path rather than file path)
        :param embedding_function: embedding function to use for the vector store
        """
        self.directory_path = vector_store_directory_path
        self.chroma_store = Chroma(persist_directory=self.directory_path, embedding_function=embedding_function)

    def save(self, chunks: List[TextChunk]) -> None:
        """
        Save chunks to Chroma vector store
        :param chunks: chunks to save
        :return: None
        """
        # Convert the chunks into text and metadata
        texts = [chunk.content for chunk in chunks]
        metadata = [
            {
                "source": chunk.document_name,
                "location": chunk.chunk_location
            }
            for chunk in chunks
        ]

        # Save into Chroma store
        self.chroma_store.add_texts(texts, metadata)

    def load(self) -> List[TextChunk]:
        """
        Load chunks from Chroma vector store
        :return: A list of TextChunk loaded from a JSON file
        """
        # Get the text from Chroma store
        chroma_dict = self.chroma_store.get()

        # Convert to text chunks
        return [
            TextChunk(document, metadata["source"], metadata["location"])
            for document, metadata in zip(chroma_dict["documents"], chroma_dict["metadatas"])
        ]


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

    # Create vector store for the chunk
    from llm import LargeLanguageModelBuilder
    embedding, llm = LargeLanguageModelBuilder.get_google_gemini_model("AIzaSyDTzDiLUvpSxZfEeEh91uQ8HHadE1VRJzg")

    vector_store = ChromaVectorStore("data/vectors/chroma", embedding_function=embedding)
    vector_store.save(chunks)
    chunks = vector_store.load()
    for chunk in chunks:
        print(chunk)
