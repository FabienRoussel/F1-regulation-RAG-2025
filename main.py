import sys
import os
import pandas as pd

from src.processing.pdf_processing import extract_pdf_text_to_dataframe


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


pdf_filename = 'FIA 2025 Formula 1 Sporting Regulations - Issue 5 - 2025-04-30.pdf'
pdf_path = os.path.join(os.path.dirname(__file__), 'data', 'pdfs', pdf_filename)
df = extract_pdf_text_to_dataframe(pdf_path)
df.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'processed', 'pdf_text.csv'), index=False)
print(df['Text'][1])