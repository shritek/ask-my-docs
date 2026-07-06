from config.settings import VectorStore


def get_vector_store(vector_store: VectorStore, persist_dir: str = None):
    """
    Returns a vector store instance based on the configuration.

    Args:
        vector_store: VectorStore enum (CHROMA or QDRANT)
        persist_dir: Directory to persist the vector store (optional)

    Returns:
        Vector store instance (Chroma or Qdrant)
    """
    if vector_store == VectorStore.CHROMA:
        from langchain_chroma import Chroma
        if persist_dir:
            return Chroma(persist_directory=persist_dir)
        return Chroma()

    elif vector_store == VectorStore.QDRANT:
        from langchain_qdrant import Qdrant
        if persist_dir:
            return Qdrant(path=persist_dir)
        return Qdrant()

    else:
        raise ValueError(f"Unsupported vector store: {vector_store}")
