import pandas as pd
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

