"""
Ollama Model Configuration
Central configuration file for all Ollama model settings.
Customize all parameters here for your project.
"""

# ============================================================================
# OLLAMA SERVER CONFIGURATION
# ============================================================================

OLLAMA_CONFIG = {
    'host': 'http://localhost:11434',  # Ollama server URL
    'timeout': 300,                     # Request timeout in seconds
    'auto_pull_models': True,           # Automatically pull models if not available
}

# ============================================================================
# CHAT/GENERATION MODEL CONFIGURATION
# ============================================================================

CHAT_MODEL_CONFIG = {
    'model_name': 'llama3.2:3b',       # Model to use for text generation
    
    # Generation parameters
    'temperature': 0.7,                 # 0.0 = deterministic, 1.0 = creative
    'top_p': 0.9,                       # Nucleus sampling threshold
    'top_k': 40,                        # Top-k sampling
    'repeat_penalty': 1.1,              # Penalize repetitive text
    'num_predict': 512,                 # Max tokens to generate
    'num_ctx': 4096,                    # Context window size
    
    # System prompt for Siddha medicine
    'system_prompt': """You are an expert in Siddha medicine, an ancient Indian system of medicine. 
Your role is to provide accurate, helpful information based on the context provided from authentic Siddha medical texts.

Guidelines:
1. Answer based primarily on the provided context
2. If the context doesn't contain relevant information, say so honestly
3. Use clear, accessible language while maintaining medical accuracy
4. Cite sources when available in the context
5. If asked about medical treatment, remind users to consult qualified practitioners
6. Maintain respect for traditional wisdom while being scientifically accurate""",
    
    # Streaming settings
    'stream': False,                    # Enable streaming responses
}

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================

EMBEDDING_MODEL_CONFIG = {
    'model_name': 'nomic-embed-text',   # Model for generating embeddings
    'dimensions': 768,                   # Embedding dimensions (auto-detected)
    'batch_size': 10,                    # Batch size for embedding generation
    'normalize': True,                   # Normalize embeddings
}

# ============================================================================
# VECTOR DATABASE CONFIGURATION
# ============================================================================

VECTORDB_CONFIG = {
    'db_path': 'data/vectordb',         # Path to ChromaDB storage
    'collection_name': 'siddha_knowledge',
    'distance_metric': 'cosine',        # cosine, l2, or ip (inner product)
    'batch_size': 100,                  # Batch size for storing vectors
}

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

RETRIEVAL_CONFIG = {
    'top_k': 10,                        # Number of chunks to retrieve (changed from 5 to 10)
    'similarity_threshold': None,       # Minimum similarity score (None = no filter)
    'rerank': False,                    # Enable re-ranking (future feature)
    'deduplicate': True,                # Remove duplicate chunks
}

# ============================================================================
# GRAPH DATABASE CONFIGURATION
# ============================================================================

GRAPHDB_CONFIG = {
    'neo4j_uri': 'bolt://localhost:7687',
    'neo4j_user': 'neo4j',
    'neo4j_password': None,             # Set via environment or parameter
    'data_path': 'data/graphdb',        # Path to store graph exports
    'embedding_model': 'nomic-embed-text',
    'traversal_depth': 2,               # Max hops for graph queries
    'vector_weight': 0.6,               # Weight for vector results in hybrid
    'graph_weight': 0.4,                # Weight for graph results in hybrid
}

# ============================================================================
# RAG MODE CONFIGURATION
# ============================================================================

RAG_MODE_CONFIG = {
    'default_mode': 'vector',           # 'vector', 'graph', or 'hybrid'
    'available_modes': ['vector', 'graph', 'hybrid'],
    'descriptions': {
        'vector': 'Fast semantic search using ChromaDB (no Neo4j required)',
        'graph': 'Relationship-based retrieval using Neo4j knowledge graph',
        'hybrid': 'Combined vector + graph for best accuracy (recommended)'
    }
}

# ============================================================================
# TEXT PROCESSING CONFIGURATION
# ============================================================================

TEXT_PROCESSING_CONFIG = {
    'chunk_size': 1000,                 # Characters per chunk
    'chunk_overlap': 200,               # Overlap between chunks
    'separator': ["\n\n", "\n", ". ", " ", ""],  # Chunking separators
    'max_chunk_size': 2000,             # Maximum chunk size
    'min_chunk_size': 100,              # Minimum chunk size
}

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRESETS = {
    'conservative': {
        'temperature': 0.3,
        'top_p': 0.8,
        'top_k': 20,
        'repeat_penalty': 1.2,
        'description': 'More focused, factual responses'
    },
    'balanced': {
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 40,
        'repeat_penalty': 1.1,
        'description': 'Balance between creativity and accuracy'
    },
    'creative': {
        'temperature': 0.9,
        'top_p': 0.95,
        'top_k': 60,
        'repeat_penalty': 1.0,
        'description': 'More creative, varied responses'
    },
    'precise': {
        'temperature': 0.1,
        'top_p': 0.7,
        'top_k': 10,
        'repeat_penalty': 1.3,
        'description': 'Most deterministic, factual responses'
    }
}

# ============================================================================
# MODEL AVAILABILITY CHECK
# ============================================================================

REQUIRED_MODELS = {
    'chat': CHAT_MODEL_CONFIG['model_name'],
    'embedding': EMBEDDING_MODEL_CONFIG['model_name']
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_preset(preset_name: str) -> dict:
    """Get a preset configuration by name."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    return PRESETS[preset_name]

def apply_preset(preset_name: str):
    """Apply a preset to the chat model configuration."""
    preset = get_preset(preset_name)
    for key, value in preset.items():
        if key != 'description' and key in CHAT_MODEL_CONFIG:
            CHAT_MODEL_CONFIG[key] = value
    return preset

def get_config_summary() -> dict:
    """Get a summary of current configuration."""
    return {
        'ollama_host': OLLAMA_CONFIG['host'],
        'chat_model': CHAT_MODEL_CONFIG['model_name'],
        'embedding_model': EMBEDDING_MODEL_CONFIG['model_name'],
        'temperature': CHAT_MODEL_CONFIG['temperature'],
        'chunk_size': TEXT_PROCESSING_CONFIG['chunk_size'],
        'top_k_retrieval': RETRIEVAL_CONFIG['top_k'],
    }

def print_config():
    """Print current configuration."""
    print("="*80)
    print("SIDDHA RAG SYSTEM - CONFIGURATION")
    print("="*80)
    print("\n🔧 Ollama Server:")
    print(f"   Host: {OLLAMA_CONFIG['host']}")
    print(f"\n🤖 Chat Model:")
    print(f"   Model: {CHAT_MODEL_CONFIG['model_name']}")
    print(f"   Temperature: {CHAT_MODEL_CONFIG['temperature']}")
    print(f"   Top-p: {CHAT_MODEL_CONFIG['top_p']}")
    print(f"   Top-k: {CHAT_MODEL_CONFIG['top_k']}")
    print(f"\n🧠 Embedding Model:")
    print(f"   Model: {EMBEDDING_MODEL_CONFIG['model_name']}")
    print(f"   Dimensions: {EMBEDDING_MODEL_CONFIG['dimensions']}")
    print(f"\n📊 Retrieval:")
    print(f"   Top-k: {RETRIEVAL_CONFIG['top_k']}")
    print(f"   Similarity threshold: {RETRIEVAL_CONFIG['similarity_threshold']}")
    print(f"\n📝 Text Processing:")
    print(f"   Chunk size: {TEXT_PROCESSING_CONFIG['chunk_size']}")
    print(f"   Chunk overlap: {TEXT_PROCESSING_CONFIG['chunk_overlap']}")
    print("="*80)

if __name__ == "__main__":
    print_config()
