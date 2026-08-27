"""Isolate compatibility workarounds needed to import the pinned Ragas release."""

import importlib
import os
import sys
import types


MISSING_VERTEXAI_MODULE = "langchain_community.chat_models.vertexai"


def _install_missing_vertexai_module() -> None:
    """Provide the removed type that Ragas 0.4.3 imports but does not use here.

    Ragas 0.4.3 imports ``ChatVertexAI`` only to detect whether an evaluator
    supports multiple completions. LangChain Community 0.4 removed that module.
    This project evaluates with Ollama, so a sentinel type preserves the Ragas
    type check without adding or downgrading unrelated Vertex AI packages.
    """
    try:
        importlib.import_module(MISSING_VERTEXAI_MODULE)
        return
    except ModuleNotFoundError as error:
        if error.name != MISSING_VERTEXAI_MODULE:
            raise

    compatibility_module = types.ModuleType(MISSING_VERTEXAI_MODULE)
    compatibility_module.ChatVertexAI = type("UnavailableChatVertexAI", (), {})
    sys.modules[MISSING_VERTEXAI_MODULE] = compatibility_module


def load_ragas_components():
    """Import Ragas only after applying the narrow upstream compatibility fix."""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    _install_missing_vertexai_module()

    import ragas
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    return {
        "version": ragas.__version__,
        "llm_factory": llm_factory,
        "embedding_factory": embedding_factory,
        "metric_classes": {
            "faithfulness": Faithfulness,
            "answer_relevancy": AnswerRelevancy,
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
        },
    }
