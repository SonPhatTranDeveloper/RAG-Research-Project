"""
Author: Trang Anh Thuan & Son Phat Tran
This file defines the various converter used to convert PDF files to Markdown files
"""
import os
from pathlib import Path
from abc import ABC, abstractmethod

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.data.data_reader_writer import FileBasedDataWriter


class MarkdownConverter(ABC):
    """
    Define a base class for a generic markdown converter
    """

    @abstractmethod
    def convert(self, input_path: str) -> str:
        """
        Convert a file in file path to a markdown file
        :param input_path: location of the input file
        :return: the markdown content (str)
        """
        return ""


class MinerUConverter(MarkdownConverter, ABC):
    def __init__(self, image_folder: str):
        """
        Define a MinerU text splitter, with image folder to save images in the document
        :param image_folder: where to save images in the documents
        """
        self.image_folder = image_folder
        self.keys = {"_pdf_type": "", "model_list": []}

    def convert(self, input_path: str) -> str:
        """
        Convert a single PDF file to markdown
        :param input_path: input file path
        :return: the file content as Markdown (str)
        """
        # Extract all the components of the path
        parts = Path(input_path)

        # Get the file name without extension
        file_name = parts.parts[-1]
        file_name_without_extension = file_name.split(".")[0]

        # Read the file as bytes
        with open(input_path, "rb") as f:
            file_as_bytes = f.read()

        # Define the image folder to save image
        file_image_folder = os.path.join(self.image_folder, file_name_without_extension)
        file_image_writer = FileBasedDataWriter(file_image_folder)

        # Create a pipeline and extract the content
        pipe = UNIPipe(file_as_bytes, self.keys, image_writer=file_image_writer)
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()
        md_content = pipe.pipe_mk_markdown(file_image_folder, drop_mode="none")

        # Return the markdown content
        return md_content
