from enum import Enum

class EmbeddingModel(Enum):
    NOMIC = "nomic"
    QWEN3 = "qwen3"
    OPENAI = "openai"

class LLMModel(Enum):
    LLAMA3_8B = "llama3.1:8b"
    LLAMA3_3B = "llama3.2:3b"

class VectorStore(Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"

CHUNKING_STRATEGY_MAPPING = {
    "recursive-500": "data/chunked_corpus_recursive-500.json",
    "recursive-1000": "data/chunked_corpus_recursive-1000.json",
    "semantic": "data/chunked_corpus_semantic.json",
}

# Defaults
DEFAULT_EMBEDDING = EmbeddingModel.NOMIC
DEFAULT_LLM = LLMModel.LLAMA3_8B
DEFAULT_VECTOR_STORE = VectorStore.CHROMA

# Use the mapping if available, otherwise default to a consistent naming pattern
CHUNKED_CORPUS_PATH = lambda strategy: CHUNKING_STRATEGY_MAPPING.get(strategy, f"data/chunked_corpus_{strategy}.json")