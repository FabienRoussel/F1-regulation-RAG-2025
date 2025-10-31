import sys
import os
import pandas as pd
import logging

from langchain_postgres import PGVector
from src.processing.pdf_processing import PDFProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


def main():
    pdf_filename = 'FIA 2025 Formula 1 Sporting Regulations - Issue 5 - 2025-04-30.pdf'
    pdf_path = os.path.join(os.path.dirname(__file__), 'data', 'pdfs', pdf_filename)
    processed_data_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')

    logger.info("Starting F1 Regulation RAG processing")

    processor = PDFProcessor(pdf_name='FIA_2025_Formula_1_Sporting_Regulations', embedding_model_name='Qwen/Qwen3-Embedding-0.6B')

    # Process PDF and extract text
    logger.info(f"Processing PDF: {pdf_filename}")
    df = processor.extract_pdf_text_to_dataframe(pdf_path)
    df.to_csv(os.path.join(processed_data_dir, 'pdf_text.csv'), index=False)
    logger.info(f"Extracted text from {len(df)} pages")

    # Extract sections
    logger.info("Extracting regulation sections...")
    df_sections = processor.extract_sections_to_dataframe('\n'.join(df['Text']))
    df_appendices = processor.extract_appendices_to_dataframe('\n'.join(df['Text']))
    df_sections.to_csv(os.path.join(processed_data_dir, 'pdf_sections.csv'), index=False)
    df_appendices.to_csv(os.path.join(processed_data_dir, 'pdf_appendices.csv'), index=False)
    df_chunks = pd.concat([df_sections, df_appendices], ignore_index=True)
    logger.info(f"Extracted {len(df_sections)} regulation sections")

    # Generate and store embeddings for sections
    logger.info("Generating and storing embeddings for sections...")
    docs = processor.create_documents_from_dataframe(df_chunks)
    logger.info(f"Created {len(docs)} Document objects")

    # See docker command above to launch a postgres instance with pgvector enabled.
    connection = "postgresql+psycopg://postgres:example@localhost:54320/mydb"  
    collection_name = "F1_Regulations"

    vector_store = PGVector(
        embeddings=processor.embedding_model,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True
    )

    # Process documents in smaller batches to avoid memory issues
    batch_size = 5  # Reduced batch size
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        batch_ids = [doc.metadata["id"] for doc in batch]
        try:
            vector_store.add_documents(batch, ids=batch_ids)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(docs) + batch_size - 1)//batch_size}")
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            # Try processing documents one by one in this batch
            for j, doc in enumerate(batch):
                try:
                    vector_store.add_documents([doc], ids=[batch_ids[j]])
                    logger.info(f"Processed document {i + j + 1}/{len(docs)}")
                except Exception as doc_error:
                    logger.error(f"Error processing document {batch_ids[j]}: {doc_error}")

    logger.info(f"Stored {len(docs)} documents in the vector store '{collection_name}'")

if __name__ == "__main__":
    main()