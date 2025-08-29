import sys
import os
import pandas as pd
import psycopg2
import logging

from psycopg2 import OperationalError

from src.processing.pdf_processing import PDFProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


def main():
    logger.info("Voici un log")
    pdf_filename = 'FIA 2025 Formula 1 Sporting Regulations - Issue 5 - 2025-04-30.pdf'
    pdf_path = os.path.join(os.path.dirname(__file__), 'data', 'pdfs', pdf_filename)
    processed_data_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')

    logger.info("Starting F1 Regulation RAG processing")

    processor = PDFProcessor(embedding_model_name='Qwen/Qwen3-Embedding-0.6B')

    # Process PDF and extract text
    logger.info(f"Processing PDF: {pdf_filename}")
    df = processor.extract_pdf_text_to_dataframe(pdf_path)
    df.to_csv(os.path.join(processed_data_dir, 'pdf_text.csv'), index=False)
    logger.info(f"Extracted text from {len(df)} pages")

    # Extract sections
    logger.info("Extracting regulation sections...")
    df_sections = pd.concat([processor.extract_sections_to_dataframe(page) for page in df['Text']], ignore_index=True)
    df_sections.to_csv(os.path.join(processed_data_dir, 'pdf_sections.csv'), index=False)
    logger.info(f"Extracted {len(df_sections)} regulation sections")

    # Generate and store embeddings for sections
    logger.info("Generating and storing embeddings for sections...")
    section_ids = processor.embed_chunks(df_sections)
    print(f"Generated embeddings for {len(section_ids)} sections")

    # Create in postgres DB
    conn = psycopg2.connect(
        database="mydb",
        user='postgres',
        password='example',
        host='localhost',
        port='5432'
    )

    conn.autocommit = True
    cursor = conn.cursor()

    with open('src/sql/create_database.sql', 'r') as fd:
        sqlFile = fd.read()
        cursor.execute(sqlFile)

    conn.commit()

    cursor.execute("\\d+ regulations;")
    print(cursor.fetchall())
    conn.close()
    

if __name__ == "__main__":
    main()