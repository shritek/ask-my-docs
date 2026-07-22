import argparse
import os
import json
import logging
from config.settings import EmbeddingModel, DEFAULT_EMBEDDING, DEFAULT_LLM, CHUNKED_CORPUS_PATH, CHUNKING_STRATEGY_MAPPING, LLMModel
from helpers.embedding_factory import get_embedder
from helpers.llm_factory import get_llm_model
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs from httpx and ollama
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)

DATA_DIR = "data"


def get_vector_store_identity(
    chunking_strategy: str,
    embedding_model: EmbeddingModel,
) -> tuple[str, str]:
    """Return an index name and path unique to a chunking/embedder pair."""
    index_key = f"{chunking_strategy}__{embedding_model.value}"
    collection_name = f"ask_my_docs_{index_key}"
    persist_dir = os.path.join(DATA_DIR, "vector_stores", index_key)
    return collection_name, persist_dir


def load_chunks(corpus_path: str) -> list[dict]:
    """Loads chunked corpus from disk."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"❌ Chunked corpus not found at {corpus_path}. Run chunker.py first.")
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def build_vector_store(chunks: list[dict], embedder, collection_name: str, persist_dir: str) -> Chroma:
    """Embeds chunks and stores them in Chroma using batching to prevent Ollama crashes."""
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Initialize an empty Chroma client with the embedder
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_dir
    )

    # Process in batches to avoid overloading Ollama (EOF 400 errors)
    batch_size = 50
    total = len(texts)
    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]
        vector_store.add_texts(texts=batch_texts, embeddings=None, metadatas=batch_metadatas)
        logger.info(f"Indexed batch {i // batch_size + 1}/{(total // batch_size) + 1} ({i + len(batch_texts)}/{total})")

    logger.info(f"✅ Vector store built: {total} chunks indexed in {collection_name}")
    return vector_store


def run(embedding_model: EmbeddingModel, chunking_strategy: str, llm_model: LLMModel, query: str):
    embedder = get_embedder(embedding_model)
    llm = get_llm_model(llm_model)
    corpus_path = CHUNKED_CORPUS_PATH(chunking_strategy)
    chunks = load_chunks(corpus_path)

    collection_name, persist_dir = get_vector_store_identity(
        chunking_strategy,
        embedding_model,
    )

    # Use the vector store if it exists, otherwise build it
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        logger.info(f"Loading existing vector store from {persist_dir}...")
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=persist_dir
        )
    else:
        vector_store = build_vector_store(chunks, embedder, collection_name, persist_dir)

    # Define retrieval tool
    @tool(response_format="content_and_artifact")
    def retrieve_context(query_text: str):
        """Retrieve information to help answer a query from the documented corpus."""
        retrieved_docs = vector_store.similarity_search(query_text, k=3)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    tools = [retrieve_context]
    tool_node = ToolNode(tools)

    # Define a proper ChatPromptTemplate for the Tool Calling Agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on provided documentation. "
                   "Use the tools provided to retrieve context from the knowledge base. "
                   "If the retrieved context does not contain relevant information, say you don't know. "
                   "Stay grounded in the documents."),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # Llama 3.1 tool calling agent logic using a simple state graph (LangGraph)
    def call_model(state: MessagesState):
        messages = prompt.invoke({"messages": state["messages"]})
        response = llm.bind_tools(tools).invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    logger.info(f"\n🔍 Query: {query.strip()}")
    print("=" * 60)

    final_state = app.invoke({"messages": [("user", query)]})
    print(final_state["messages"][-1].content)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic RAG pipeline")
    parser.add_argument(
        "query",
        help="The question you want to ask the RAG system"
    )
    parser.add_argument(
        "--chunking_strategy",
        choices=list(CHUNKING_STRATEGY_MAPPING.keys()),
        required=True,
        help="Chunking strategy to use for retrieval"
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=DEFAULT_EMBEDDING.value,
        choices=[e.value for e in EmbeddingModel]
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=DEFAULT_LLM.value,
        choices=[e.value for e in LLMModel]
    )
    args = parser.parse_args()

    run(
        embedding_model=EmbeddingModel(args.embedder),
        chunking_strategy=args.chunking_strategy,
        llm_model=LLMModel(args.llm),
        query=args.query
    )
