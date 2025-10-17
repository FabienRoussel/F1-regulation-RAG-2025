from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from generation.models.ollama_model import OllamaModel


def main():
    """Main function to query F1 regulations."""

    # Initialize and pull Ollama model
    print("=" * 60)
    print("F1 Regulations Query System")
    print("=" * 60)

    ollama = OllamaModel(base_url="http://localhost:11434", model_name="gemma3:4b")
    print("\nStep 1: Pulling Ollama model...")

    if not ollama.is_model_available():
        ollama.pull_model()
    else:
        print("Model already available, skipping download.")

    # Ask user for a question
    print("\n" + "=" * 60)
    print("Step 2: Enter your question about F1 regulations")
    print("=" * 60)
    question = input("\nYour question: ").strip()

    if not question:
        print("No question provided. Exiting.")
        return

    print(f"\nReceived question: {question}")

    # Initialize embedding model
    print("\n" + "=" * 60)
    print("Step 3: Embedding your question with PgVector")
    print("=" * 60)

    embedding_model = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'mps'}
    )

    # Connect to PgVector database
    connection = "postgresql+psycopg://postgres:example@localhost:54320/mydb"
    collection_name = "F1_Regulations"

    print(f"\nConnecting to database: {connection}")
    print(f"Collection name: {collection_name}")

    # Initialize PgVector store and retriever
    vectorstore = PGVector(
        embeddings=embedding_model,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    # Create base retriever
    print("\n" + "=" * 60)
    print("Step 4: Creating retriever with reranker")
    print("=" * 60)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}  # Retrieve more docs for reranking
    )

    # Initialize cross-encoder reranker
    print("\nInitializing cross-encoder reranker...")
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)

    # Create compression retriever with reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    # Query the retriever
    print("\n" + "=" * 60)
    print("Step 5: Retrieving relevant documents")
    print("=" * 60)

    print(f"\nSearching for documents related to: {question}")
    docs = compression_retriever.invoke(question)

    print(f"\nFound {len(docs)} relevant documents after reranking:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(f"Content preview: {doc.page_content[:200]}...")
        if hasattr(doc, 'metadata'):
            print(f"Metadata: {doc.metadata}")

    # Generate answer using Ollama
    print("\n" + "=" * 60)
    print("Step 6: Generating answer with Ollama")
    print("=" * 60)

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Based on the following F1 regulation context, answer the question.

    Context:
    {context}

    Question: {question}

    Answer:"""

    print("\nGenerating answer...")
    answer = ollama.generate(prompt)

    print("\n" + "=" * 60)
    print("Answer:")
    print("=" * 60)
    print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
