from config.settings import LLMModel
from langchain.chat_models import init_chat_model


def get_llm_model(model: LLMModel):
    if model == LLMModel.LLAMA3_8B:
        return init_chat_model(
            "llama3.1:8b",
            model_provider="ollama"
        )
    elif model == LLMModel.LLAMA3_3B:
        return init_chat_model(
            "llama3.2:3b",
            model_provider="ollama"
        )
    elif model == LLMModel.GEMMA4_31B:
        return init_chat_model(
            "gemma4:31b-mlx",
            model_provider="ollama"
        )
    else:
        raise ValueError(f"Unsupported LLM model: {model}")
