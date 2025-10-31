import sys
import os
# typing.Optional not used

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from generation.retriever.tools import document_retriever_F1
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage


def build_prompt_from_docs(docs, question: str) -> str:
    """Create a single prompt by concatenating retrieved docs and the question."""
    context = "\n\n".join([getattr(d, 'page_content', str(d)) for d in docs])
    prompt = f"""Based on the following F1 regulation context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def conversational_cli(
    retriever,
    chat_model_name: str = "gemma3:4b",
    base_url: str = "http://localhost:11434",
):
    """Start a conversational CLI loop using only LangChain's ChatOllama model.

    This function will exit with an explanatory message if ChatOllama cannot be
    instantiated (so the user must install the LangChain Ollama integration).
    """
    print("=" * 60)
    print("F1 Regulations Conversational Agent (type 'exit' to quit)")
    print("Commands: 'help', 'exit'\n")

    # Instantiate ChatOllama; if this fails we must abort (user requested no OllamaModel fallback)
    try:
        chat_model = ChatOllama(model=chat_model_name, base_url=base_url)
        print(f"Using ChatOllama model: {chat_model_name} at {base_url}")
    except Exception as e:
        print("Could not initialize ChatOllama. Ensure langchain and the ollama integration are installed and Ollama is running.")
        print(f"Error: {e}")
        return

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if user_input.lower() == "help":
            print("Enter a question about F1 regulations. Type 'exit' to quit.")
            continue

        # Retrieve documents for the user query
        print("\nRetrieving relevant documents...")
        docs = retriever.get_relevant_documents(user_input)
        print(f"Found {len(docs)} documents (showing snippets):")
        for i, d in enumerate(docs[:3], 1):
            snippet = getattr(d, 'page_content', '')[:300]
            meta = getattr(d, 'metadata', {})
            print(f"\n[{i}] {snippet}...\n   metadata={meta}")

        prompt = build_prompt_from_docs(docs, user_input)

        print("\nGenerating answer with ChatOllama...")
        try:
            # Use predict_messages for a single user message and get a BaseMessage response
            resp = chat_model.predict_messages([HumanMessage(content=prompt)])
            answer = getattr(resp, 'content', str(resp))
        except Exception as e:
            answer = f"ChatOllama generation failed: {e}"

        print("\n" + "=" * 60)
        print("Agent:\n")
        print(answer)
        print("\n" + "=" * 60)


def main():
    print("=" * 60)
    print("F1 Regulations Conversational RAG")
    print("=" * 60)

    # Create the document retriever using the factory
    retriever = document_retriever_F1()

    # Start conversational CLI using ChatOllama only
    conversational_cli(retriever=retriever, chat_model_name="gemma3:4b", base_url="http://localhost:11434")


if __name__ == "__main__":
    main()

