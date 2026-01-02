"""
Configuration for Graph Database
Centralized settings for Neo4j connection and graph operations.
"""

import os
import sys
from pathlib import Path

# Try to import from models/config.py
try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT / 'models'))
    from config import GRAPHDB_CONFIG as MAIN_CONFIG
    
    # Use settings from main config
    NEO4J_URI = MAIN_CONFIG.get('neo4j_uri', 'bolt://localhost:7687')
    NEO4J_USER = MAIN_CONFIG.get('neo4j_user', 'neo4j')
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or MAIN_CONFIG.get('neo4j_password', 'password')
    GRAPH_DATA_PATH = MAIN_CONFIG.get('data_path', 'data/graphdb')
except ImportError:
    # Fallback defaults
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GRAPH_DATA_PATH = "data/graphdb"

# Embedding Settings
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768

# Schema Settings
ENTITY_TYPES = [
    "Herb",
    "Medicine", 
    "Disease",
    "Symptom",
    "Treatment",
    "Ingredient",
    "SideEffect",
    "BodyPart",
    "Document"
]

RELATIONSHIP_TYPES = [
    "TREATS",
    "RELIEVES",
    "CONTAINS",
    "HAS_SYMPTOM",
    "AFFECTS",
    "HAS_SIDE_EFFECT",
    "CONTRAINDICATED_WITH",
    "INTERACTS_WITH",
    "MENTIONS",
    "SUPPORTS"
]

# Extraction Settings
EXTRACTION_BATCH_SIZE = 10
MAX_TRIPLES_PER_CHUNK = 20
MIN_CONFIDENCE_THRESHOLD = 0.5

# Retrieval Settings
DEFAULT_TOP_K = 5
GRAPH_TRAVERSAL_DEPTH = 2  # Max hops for graph queries
HYBRID_WEIGHT_VECTOR = 0.6  # Weight for vector results in hybrid mode
HYBRID_WEIGHT_GRAPH = 0.4   # Weight for graph results in hybrid mode

# File Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCHEMA_FILE = PROJECT_ROOT / "src" / "graph" / "schema.cypher"
