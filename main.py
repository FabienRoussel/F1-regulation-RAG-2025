import sys
import os
import pandas as pd

from src.processing.pdf_processing import extract_pdf_text_to_dataframe, extract_sections_to_dataframe


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


pdf_filename = 'FIA 2025 Formula 1 Sporting Regulations - Issue 5 - 2025-04-30.pdf'
pdf_path = os.path.join(os.path.dirname(__file__), 'data', 'pdfs', pdf_filename)
processed_data_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')

# Splitting the PDF into pages and extracting text
df = extract_pdf_text_to_dataframe(pdf_path)
df.to_csv(os.path.join(processed_data_dir, 'pdf_text.csv'), index=False)

# Extracting each section from the text and saving to a DataFrame, except appendices
df_sections = pd.concat([extract_sections_to_dataframe(page) for page in df['Text']], ignore_index=True)
df_sections.to_csv(os.path.join(processed_data_dir, 'pdf_sections.csv'), index=False)