# RAG Research Project

## About

This repository contains the code for a Retrieval-Augmented Generation Research project for RMIT University.

## How to run

### Step 1: Create data folder (OPTIONAL)

Create a folder called ```data``` in the root folder. In that folder:
- Create ```data/pdfs``` and place your PDF files in that folder.
- Create ```data/markdowns``` that contains ```data/markdowns/images``` and ```data/markdowns/text``` to save the result
of markdown converter.
- Create ```data/vectors``` to save the text chunks

Note: 
- You can always use a different folder structure. Just make sure to reflect it by changing the paths in the
relevant Python files (mentioned in the steps below).
- If you use the above folder structure, you don't have to change the paths (just change the API keys).

### Step 2: Convert PDFs to Markdown files

For the converter step, we use the MagicPDF package from MinerU. First, you should start by running the set-up file in
the script folder. This will create the ```magic-pdf.json``` config file and download relevant OCR/Deep Learning models
for the package to function properly.

```bash
python scripts/magic_pdf_setup.py
```

Then replace all the paths in ```markdown_converter/__init__.py``` folder with your appropriate paths.
Finally, from the root folder, run
```bash
python markdown_converter/__init__.py
```
to convert all the PDF files to Markdown files.

### Step 3: Run the RAG pipeline
Replace the paths and API keys in ```pipeline/__init__.py``` file with your appropriate keys and keys.
Also, replace the question to one of your liking. Finally, from the root folder, run
```bash
python markdown_converter/__init__.py
```
to obtain the result.

### Step 4 (Optional): Run the web application
To be added in the future.