from config.settings import EmbeddingModel

def get_embedder(model: EmbeddingModel):
    if model == EmbeddingModel.NOMIC:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")
    
    elif model == EmbeddingModel.QWEN3:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-8B"
        )
    
    elif model == EmbeddingModel.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    
    else:
        raise ValueError(f"Unsupported embedding model: {model}")