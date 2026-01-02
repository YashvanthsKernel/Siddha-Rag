"""
Models Package
Centralized model management for the Siddha RAG system.
"""

from .config import (
    OLLAMA_CONFIG,
    CHAT_MODEL_CONFIG,
    EMBEDDING_MODEL_CONFIG,
    VECTORDB_CONFIG,
    RETRIEVAL_CONFIG,
    TEXT_PROCESSING_CONFIG,
    PRESETS,
    get_preset,
    apply_preset,
    get_config_summary,
    print_config
)

from .model_manager import OllamaManager

__all__ = [
    'OllamaManager',
    'OLLAMA_CONFIG',
    'CHAT_MODEL_CONFIG',
    'EMBEDDING_MODEL_CONFIG',
    'VECTORDB_CONFIG',
    'RETRIEVAL_CONFIG',
    'TEXT_PROCESSING_CONFIG',
    'PRESETS',
    'get_preset',
    'apply_preset',
    'get_config_summary',
    'print_config'
]
