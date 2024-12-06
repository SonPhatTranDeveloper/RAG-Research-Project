"""
Author: Son Phat Tran
This file contains various LLMs used in the RAG pipeline
"""
from typing import Tuple
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaLLM, OllamaEmbeddings


class LargeLanguageModelBuilder:
    """
    Define a builder for various LLMs model to use in LangChain pipeline
    """

    @staticmethod
    def get_open_ai_model(api_key: str,
                          model_name: str = "text-embedding-3-small",
                          embedding_name: str = "gpt-4o-mini") -> Tuple[OpenAIEmbeddings, ChatOpenAI]:

        # Create embedding and llm models
        embedding = OpenAIEmbeddings(model=embedding_name, api_key=api_key)
        llm = ChatOpenAI(model=model_name, api_key=api_key)

        # Return embedding and llm
        return embedding, llm

    @staticmethod
    def get_google_gemini_model(api_key: str,
                                model_name: str = "gemini-1.5-flash",
                                embedding_name: str = "models/embedding-001") -> Tuple[GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI]:

        # Set the key
        os.environ["GOOGLE_API_KEY"] = api_key

        # Create embedding and llm models
        embedding = GoogleGenerativeAIEmbeddings(model=embedding_name)
        llm = ChatGoogleGenerativeAI(model=model_name)

        # Return embedding and llm
        return embedding, llm

    @staticmethod
    def get_ollama_model(api_key: str,
                         model_name: str = "llama3.2",
                         embedding_name: str = "llama3.2") -> Tuple[OllamaEmbeddings, OllamaLLM]:

        # Create embedding and llm models
        embedding = OllamaEmbeddings(model=embedding_name, base_url=api_key)
        llm = OllamaLLM(model=model_name, base_url=api_key)

        # Return embedding and llm
        return embedding, llm


if __name__ == "__main__":
    pass
