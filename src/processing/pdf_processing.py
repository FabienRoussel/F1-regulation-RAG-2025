import pandas as pd
import re

from langchain_community.document_loaders import PyMuPDFLoader
from src.utils.utils import remove_footer

def extract_pdf_text_to_dataframe(pdf_path: str) -> pd.DataFrame:
    """
    Extracts text from a PDF file and returns it in a pandas DataFrame.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted text, with each page as a row.
    """
    try:
        
        # Load the document
        loader = PyMuPDFLoader(pdf_path,extract_tables='markdown')
        documents = loader.load_and_split()

        pages_text = [remove_footer(doc.page_content) for doc in documents]

        df = pd.DataFrame({'Page': range(1, len(pages_text) + 1), 'Text': pages_text})
        return df


    except Exception as e:
        raise RuntimeError(f"Error processing PDF file: {e}")

def extract_sections_to_dataframe(text: str) -> pd.DataFrame:
    """
    Extracts sections from the text based on patterns like 'NUMBER) A TITLE'
    and returns a DataFrame with 'title' and 'chunk' columns.

    Args:
        text (str): The text to process.

    Returns:
        pd.DataFrame: A DataFrame with 'title' and 'chunk' columns.
    """
    # Updated regex to handle cases like "VIRTUAL SAFETY CAR (VSC)"
    # and exclude titles with letters before the number
    section_pattern = r"(?<![A-Z])(\d+\)\s+(?:[A-Z ,\n\-\(\)É]+))\n"
    matches = re.split(section_pattern, text)

    titles = []
    chunks = []

    for i in range(1, len(matches), 2):
        titles.append(matches[i].replace('\n', ' ').strip())  # Clean multi-line titles
        chunks.append(matches[i + 1].strip())

    processed_df = pd.DataFrame({'title': titles, 'chunk': chunks})
    processed_df[['title', 'chunk']] = processed_df[['title', 'chunk']].map(lambda x: x.replace('\n', ' ').strip())

    return processed_df
