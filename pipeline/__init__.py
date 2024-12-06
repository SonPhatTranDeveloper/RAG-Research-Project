"""
Author: Son Phat Tran
This file contains the logic for the RAG pipeline
"""
from typing import Dict, List, Any

from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from text_segmentor import TextSegmentor, MarkDownHeaderSegmentor
from vector_store import VectorStore, JsonVectorStore
from vector_store.folder import persist_vectors
from document_retriever import DocumentRetrieverBuilder, BM25RetrieverBuilder, ChromaRetrieverBuilder
from llm import LargeLanguageModelBuilder


class RAGAnswer:
    def __init__(self, answer,
                 keyword_context,
                 keyword_metadata,
                 semantic_context,
                 semantic_metadata):
        """
        This class represents the answer for the RAG
        :param answer
        :param keyword_context
        :param keyword_metadata
        :param semantic_context
        :param semantic_metadata
        """
        self.answer = answer
        self.keyword_context = keyword_context
        self.keyword_metadata = keyword_metadata
        self.semantic_context = semantic_context
        self.semantic_metadata = semantic_metadata


def combine_results(results: Dict) -> List:
    """
    Combine the results from two retrievers
    :param results: dictionary containing the results
    :return: combine result
    """
    return results["keyword"] + results["semantic"]


def format_documents(docs: List[Document]) -> str:
    """
    Format the documents by adding new lines between them
    :param docs: the document
    :return:
    """
    return "\n\n".join(doc.page_content for doc in docs)


class RAGPipeline:
    def __init__(self,
                 document_folder: str,
                 llm_model: Any, embedding_model: Any,
                 text_segmentor: TextSegmentor,
                 vector_store: VectorStore,
                 prompt_template: Any,
                 keyword_retriever_builder: DocumentRetrieverBuilder,
                 semantic_retriever_builder: DocumentRetrieverBuilder,
                 build_vector_store=False) -> None:
        """
        Create a RAG pipeline using various components
        :param document_folder: the folder that contains the document
        :param llm_model: large language model for semantic retriever
        :param embedding_model: embedding for semantic retriever
        :param text_segmentor: text segmenting strategy
        :param vector_store: text chunk store
        :param prompt_template: the prompt used for RAG
        :param keyword_retriever_builder: builder for keyword-based document retriever
        :param semantic_retriever_builder: builder for semantic-based document retriever
        :param build_vector_store: whether to rebuild vector store or to use existing ones
        """
        # Save the document folder
        self.prompt = prompt_template
        self.document_folder = document_folder

        # Save LLMs and embeddings
        self.llm = llm_model
        self.embedding = embedding_model

        # Save the text segmentor
        self.text_segmentor = text_segmentor

        # Save the vector store
        self.vector_store = vector_store

        # Either load the document from vector store or to create it from scratch
        if build_vector_store:
            all_chunks = persist_vectors(
                document_folder,
                text_segmentor,
                vector_store
            )
        else:
            all_chunks = vector_store.load()

        # Build the keyword retriever
        self.keyword_retriever = keyword_retriever_builder.build(all_chunks)
        self.semantic_retriever = semantic_retriever_builder.build(all_chunks)
        self.parallel_retriever = RunnableParallel({
            "keyword": self.keyword_retriever,
            "semantic": self.semantic_retriever
        })

        # Create the pipeline
        self.rag_chain = (
                {"context": self.parallel_retriever | combine_results | format_documents,
                 "question": RunnablePassthrough()}
                | prompt_template
                | llm_model
                | StrOutputParser()
        )

    def invoke(self, question) -> RAGAnswer:
        # Find the relevant documents
        docs = self.parallel_retriever.invoke(question)

        # Separate context and metadata for each retriever
        semantic_docs = docs["keyword"]
        text_docs = docs["semantic"]

        # Format the semantic context and metadata
        semantic_context = "\n".join([doc.page_content for doc in semantic_docs])
        semantic_metadata = "\n".join(
            [f"{doc.metadata['source']} (Chunk: {doc.metadata['location']})" for doc in semantic_docs])

        # Format the keyword context and metadata
        keyword_context = "\n".join([doc.page_content for doc in text_docs])
        keyword_metadata = "\n".join(
            [f"{doc.metadata['source']} (Chunk: {doc.metadata['location']})" for doc in text_docs])

        # Get the answer from the chain
        answer = self.rag_chain.invoke(question)

        # Return
        return RAGAnswer(
            answer,
            keyword_context,
            keyword_metadata,
            semantic_context,
            semantic_metadata
        )


if __name__ == "__main__":
    # Define the various parameters
    # This is the output folder that contain the converted Markdown files
    doc_folder = "data/markdowns/text"

    # Create the LLMs and Embeddings using your API key
    # Remember to replace with your Google Gemini API Key
    embedding, llm = LargeLanguageModelBuilder.get_google_gemini_model("[YOUR TOKEN GOES HERE]")

    # Create the text segmentor
    # More will be created in the future
    segmentor = MarkDownHeaderSegmentor()

    # Create a store for text chunks
    # Replace with your appropriate folder
    text_store = JsonVectorStore("data/vectors/json/data.json")

    # Create prompt template
    prompt = hub.pull("rlm/rag-prompt")

    # Create BM25 keyword retriever and Chroma semantic retriever
    keyword_builder = BM25RetrieverBuilder(k=1)
    semantic_builder = ChromaRetrieverBuilder(embedding, k=1)

    # Create RAG
    rag_pipeline = RAGPipeline(
        document_folder=doc_folder,
        llm_model=llm, embedding_model=embedding,
        text_segmentor=segmentor,
        vector_store=text_store,
        prompt_template=prompt,
        keyword_retriever_builder=keyword_builder,
        semantic_retriever_builder=semantic_builder,
        build_vector_store=True
    )

    # Get the result
    # Replace with your question
    result = rag_pipeline.invoke("What is SEC")
    print(result.answer)
