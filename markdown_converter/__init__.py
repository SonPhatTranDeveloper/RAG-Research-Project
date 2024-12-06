from markdown_converter.folder_converter import convert_folder
from markdown_converter.file_converter import MinerUConverter


if __name__ == "__main__":
    # Convert a simple file
    # This is the folder that contains PDFs
    pdf_input_path = "data/pdfs"

    # This is the output Markdown folder
    markdown_output_path = "data/markdowns/text"

    # This is the output folder for images
    image_output_path = "data/markdowns/images"

    # Convert an entire folder
    convert_folder(
        input_path=pdf_input_path,
        output_path=markdown_output_path,
        file_converter=MinerUConverter(image_output_path)
    )

