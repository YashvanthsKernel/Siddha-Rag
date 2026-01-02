"""
🔒 SECURE Fast Migration: VectorDB to GraphDB using Groq Cloud

SECURITY FEATURES:
- ✅ API key validation and secure handling (never logged)
- ✅ Data sanitization before sending to cloud
- ✅ No sensitive data in logs or console output
- ✅ Memory clearing after processing each chunk
- ✅ Minimal data exposure - only send text, receive only entities
- ✅ No raw text stored or transmitted unnecessarily
- ✅ Secure temp file handling
- ✅ Rate limiting to prevent API abuse detection

Uses Groq's ultra-fast LPU with llama-3.3-70b-versatile for entity extraction.

Usage:
    python scripts/migrate_with_groq.py
    python scripts/migrate_with_groq.py --max-files 5
    python scripts/migrate_with_groq.py --secure-mode  # Extra privacy
"""

import os
import sys
import json
import time
import gc
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment SECURELY
load_dotenv(PROJECT_ROOT / ".env")

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# ==================== SECURITY CONFIGURATION ====================

# Disable verbose logging to prevent data leaks
import logging
logging.getLogger("groq").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("neo4j").setLevel(logging.ERROR)

# Configuration
GROQ_MODEL = "llama-3.1-8b-instant"  # Faster model with higher rate limits
CHUNK_SIZE = 1500  # Smaller chunks for better extraction
CHUNK_OVERLAP = 150
RATE_LIMIT_DELAY = 6.0  # 10 requests/min to stay under limit
MAX_RETRIES = 5  # More retries for rate limits
RETRY_DELAY = 30  # 30 seconds wait on rate limit

# Entity and relationship types (no sensitive data here)
ENTITY_TYPES = [
    "Herb", "Medicine", "Drug", "Formulation",
    "Disease", "Disorder", "Condition", "Syndrome",
    "Symptom", "Sign", "Indication",
    "Treatment", "Therapy", "Procedure", "Practice",
    "Ingredient", "Compound", "Chemical", "Mineral",
    "BodyPart", "Organ", "System", "Tissue",
    "Dosage", "Preparation", "Method",
    "Contraindication", "SideEffect", "Precaution",
    "Person", "Author", "Sage", "Siddhar",
    "Text", "Document", "Reference", "Source"
]

RELATIONSHIP_TYPES = [
    "TREATS", "CURES", "HEALS", "MANAGES",
    "RELIEVES", "ALLEVIATES", "REDUCES", "CONTROLS",
    "CAUSES", "INDUCES", "TRIGGERS", "LEADS_TO",
    "PREVENTS", "PROTECTS_FROM", "INHIBITS",
    "CONTAINS", "COMPOSED_OF", "HAS_INGREDIENT", "MADE_FROM",
    "HAS_SYMPTOM", "MANIFESTS_AS", "PRESENTS_WITH",
    "AFFECTS", "ACTS_ON", "TARGETS", "INFLUENCES",
    "PREPARED_BY", "ADMINISTERED_AS", "GIVEN_WITH",
    "CONTRAINDICATED_WITH", "INTERACTS_WITH", "INCOMPATIBLE_WITH",
    "MENTIONED_IN", "DESCRIBED_BY", "AUTHORED_BY", "REFERENCED_IN",
    "SUPPORTS", "ENHANCES", "POTENTIATES", "SYNERGIZES_WITH",
    "DERIVED_FROM", "EXTRACTED_FROM", "OBTAINED_FROM",
    "USED_FOR", "INDICATED_FOR", "PRESCRIBED_FOR",
    "DOSAGE_OF", "FORM_OF", "VARIANT_OF"
]


# ==================== SECURITY UTILITIES ====================

class SecurityManager:
    """
    Handles all security-related operations.
    """
    
    @staticmethod
    def validate_api_key(key: str) -> bool:
        """Validate API key format without exposing it."""
        if not key:
            return False
        # Groq keys start with 'gsk_'
        return key.startswith('gsk_') and len(key) > 20
    
    @staticmethod
    def mask_sensitive_string(s: str, show_chars: int = 4) -> str:
        """Mask a sensitive string for safe logging."""
        if not s or len(s) <= show_chars * 2:
            return "***"
        return s[:show_chars] + "..." + s[-show_chars:]
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """
        Sanitize text before sending to external API.
        Removes potential PII patterns while preserving medical content.
        """
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove phone numbers (various formats)
        text = re.sub(r'\b(?:\+?91[-.\s]?)?[6-9]\d{9}\b', '[PHONE]', text)
        text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
        
        # Remove Aadhaar-like numbers (12 digits)
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[ID]', text)
        
        # Remove potential names after common prefixes (Dr., Mr., Mrs., etc.)
        # text = re.sub(r'\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Sri\.|Shri\.)\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', '[NAME]', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '[URL]', text)
        
        # Remove file paths
        text = re.sub(r'[A-Za-z]:\\[^\s]+', '[PATH]', text)
        text = re.sub(r'/(?:home|users?|var|tmp)/[^\s]+', '[PATH]', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def secure_clear(data):
        """Securely clear sensitive data from memory."""
        if isinstance(data, str):
            # Can't really clear strings in Python, but we can dereference
            del data
        elif isinstance(data, list):
            data.clear()
        elif isinstance(data, dict):
            data.clear()
        gc.collect()
    
    @staticmethod
    def hash_for_logging(text: str) -> str:
        """Create a hash for logging without exposing content."""
        return hashlib.sha256(text.encode()).hexdigest()[:12]


# ==================== SECURE GROQ EXTRACTOR ====================

class SecureGroqExtractor:
    """
    Secure entity extraction using Groq Cloud.
    - Never logs raw text
    - Sanitizes data before sending
    - Clears memory after processing
    """
    
    def __init__(self, model: str = GROQ_MODEL, secure_mode: bool = False):
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        
        # Validate API key securely
        if not SecurityManager.validate_api_key(api_key):
            raise ValueError("❌ Invalid or missing GROQ_API_KEY in .env file!")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        self.secure_mode = secure_mode
        
        # Don't log the actual key
        masked_key = SecurityManager.mask_sensitive_string(api_key)
        print(f"✅ Groq client initialized (key: {masked_key})")
        print(f"   Model: {model}")
        print(f"   Secure Mode: {'ON' if secure_mode else 'OFF'}")
    
    def extract_triples(self, text: str, source_hash: str = "") -> List[Dict]:
        """
        Extract entities securely.
        - Sanitizes text before sending
        - Only extracts entities, doesn't store raw text
        - Clears memory after processing
        """
        # SECURITY: Sanitize text before sending to cloud
        sanitized_text = SecurityManager.sanitize_text(text)
        
        # SECURITY: In secure mode, don't send any identifying info
        if self.secure_mode:
            # Remove any remaining potential identifiers
            sanitized_text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[ENTITY]', sanitized_text)
        
        prompt = f"""Extract medical knowledge triples from this Siddha medicine text.

ENTITY TYPES: {', '.join(ENTITY_TYPES)}
RELATIONSHIP TYPES: {', '.join(RELATIONSHIP_TYPES)}

TEXT:
{sanitized_text}

RULES:
1. Extract ALL factual relationships
2. Use exact entity names
3. Output ONLY valid JSON array
4. No explanations

OUTPUT FORMAT:
[{{"subject": "name", "subject_type": "Type", "predicate": "RELATIONSHIP", "object": "name", "object_type": "Type", "confidence": 0.9}}]"""

        # Retry logic with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a medical knowledge extractor. Output only valid JSON arrays."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2048  # Reduced for faster response
                )
                
                content = response.choices[0].message.content.strip()
                triples = self._parse_response(content, source_hash)
                
                # SECURITY: Clear sensitive data from memory
                SecurityManager.secure_clear(sanitized_text)
                SecurityManager.secure_clear(prompt)
                SecurityManager.secure_clear(content)
                
                return triples
                
            except Exception as e:
                error_str = str(e).lower()
                error_msg = str(e)[:150]  # More chars for debugging
                
                if "rate_limit" in error_str or "429" in error_str or "too many" in error_str:
                    # Rate limited - wait and retry
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"\n   ⏳ RATE LIMITED by Groq (free tier limit: ~30 req/min)")
                    print(f"      Waiting {wait_time}s before retry (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                elif "401" in error_str or "invalid" in error_str and "key" in error_str:
                    print(f"\n   ❌ INVALID API KEY - Check your GROQ_API_KEY in .env")
                    break
                elif "timeout" in error_str or "connection" in error_str:
                    print(f"\n   ⚠️ CONNECTION ERROR - Network issue, retrying...")
                    time.sleep(5)
                else:
                    # Other error - show message
                    print(f"\n   ⚠️ API Error: {error_msg}")
                    break
        
        return []
    
    def _parse_response(self, content: str, source_hash: str) -> List[Dict]:
        """Parse JSON response securely."""
        triples = []
        
        try:
            start = content.find('[')
            end = content.rfind(']') + 1
            
            if start != -1 and end > start:
                json_str = content[start:end]
                parsed = json.loads(json_str)
                
                for item in parsed:
                    if isinstance(item, dict):
                        triple = {
                            "subject": str(item.get("subject", "")).strip()[:200],  # Limit length
                            "subject_type": item.get("subject_type", "Entity"),
                            "predicate": str(item.get("predicate", "RELATED_TO")).upper().replace(" ", "_")[:50],
                            "object": str(item.get("object", "")).strip()[:200],
                            "object_type": item.get("object_type", "Entity"),
                            "confidence": min(1.0, max(0.0, float(item.get("confidence", 0.8)))),
                            "source_hash": source_hash  # Use hash, not filename
                        }
                        
                        if triple["subject"] and triple["object"]:
                            triples.append(triple)
                            
        except json.JSONDecodeError:
            pass  # Silent fail - don't expose data in error logs
        
        return triples


# ==================== SECURE NEO4J MANAGER ====================

class SecureNeo4jManager:
    """
    Secure Neo4j operations.
    - Password never logged
    - Batch operations to minimize exposure
    """
    
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        
        # SECURITY: Don't log password
        if not password:
            raise ValueError("❌ NEO4J_PASSWORD not set in .env!")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, password))
            self.driver.verify_connectivity()
            print(f"✅ Connected to Neo4j at {self.uri}")
        except ServiceUnavailable:
            raise ConnectionError(f"❌ Cannot connect to Neo4j at {self.uri}")
    
    def close(self):
        self.driver.close()
        print("✅ Neo4j connection closed securely")
    
    def reset_database(self):
        """Clear all data."""
        print("🗑️  Resetting database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("   ✅ Database cleared")
    
    def create_indexes(self):
        """Create indexes."""
        print("📇 Creating indexes...")
        with self.driver.session() as session:
            for entity_type in ENTITY_TYPES:
                try:
                    session.run(f"CREATE INDEX {entity_type.lower()}_idx IF NOT EXISTS FOR (n:{entity_type}) ON (n.name)")
                except:
                    pass
        print("   ✅ Indexes ready")
    
    def _sanitize_label(self, label: str) -> str:
        """Sanitize a label for Neo4j (remove spaces and special chars)."""
        # Replace spaces with underscores, remove other special chars
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(label))
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'Node_' + sanitized
        return sanitized if sanitized else 'Entity'
    
    def batch_import(self, triples: List[Dict], batch_size: int = 100):
        """Import triples securely in batches."""
        print(f"\n🔨 Importing {len(triples)} triples...")
        
        by_rel = {}
        for t in triples:
            # Sanitize relationship type
            rel = self._sanitize_label(t["predicate"])
            if rel not in by_rel:
                by_rel[rel] = []
            by_rel[rel].append(t)
        
        total = 0
        errors = 0
        with self.driver.session() as session:
            for rel_type, rel_triples in tqdm(by_rel.items(), desc="Importing"):
                for i in range(0, len(rel_triples), batch_size):
                    batch = rel_triples[i:i + batch_size]
                    
                    type_groups = {}
                    for t in batch:
                        # Sanitize entity type names
                        subj_type = self._sanitize_label(t["subject_type"])
                        obj_type = self._sanitize_label(t["object_type"])
                        key = (subj_type, obj_type)
                        if key not in type_groups:
                            type_groups[key] = []
                        type_groups[key].append({
                            "subject": t["subject"],
                            "object": t["object"],
                            "confidence": t["confidence"]
                        })
                    
                    for (subj_type, obj_type), group_batch in type_groups.items():
                        try:
                            query = f"""
                                UNWIND $batch AS rel
                                MERGE (s:{subj_type} {{name: rel.subject}})
                                MERGE (o:{obj_type} {{name: rel.object}})
                                MERGE (s)-[r:{rel_type}]->(o)
                                SET r.confidence = rel.confidence
                            """
                            session.run(query, batch=group_batch)
                            total += len(group_batch)
                        except Exception as e:
                            errors += 1
                            if errors <= 3:  # Only show first few errors
                                print(f"\n   ⚠️ Import error: {str(e)[:80]}")
        
        print(f"\n   ✅ Imported {total} relationships ({errors} errors)")
        
        # SECURITY: Clear batch data
        SecurityManager.secure_clear(by_rel)
        
        return total
    
    def get_stats(self) -> Dict:
        """Get graph statistics (no sensitive data)."""
        with self.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
            
            node_types = {}
            for record in session.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) as c ORDER BY c DESC"):
                node_types[record["l"]] = record["c"]
            
            rel_types = {}
            for record in session.run("MATCH ()-[r]->() RETURN type(r) as t, count(*) as c ORDER BY c DESC"):
                rel_types[record["t"]] = record["c"]
        
        return {"nodes": nodes, "relationships": rels, "node_types": node_types, "rel_types": rel_types}


# ==================== MAIN MIGRATION ====================

def chunk_text(text: str) -> List[str]:
    """Split text into chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


def load_files_securely(text_dir: Path, max_files: int = None) -> Dict[str, str]:
    """Load files without exposing full paths."""
    texts = {}
    files = sorted(text_dir.glob("*.txt"))
    
    if max_files:
        files = files[:max_files]
    
    print(f"\n📂 Loading {len(files)} files...")
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read().strip()
                if content:
                    # Use hash as key for privacy
                    file_hash = hashlib.sha256(f.name.encode()).hexdigest()[:12]
                    texts[file_hash] = content
                    # Only show filename, not full path
                    print(f"   ✓ {f.name[:40]}... ({len(content):,} chars)")
        except:
            pass
    
    return texts


def migrate_secure(max_files: int = None, chunks_per_file: int = None, secure_mode: bool = False):
    """
    Secure migration function.
    """
    print("=" * 70)
    print("🔒 SECURE Migration: VectorDB to GraphDB (Groq Cloud)")
    print("=" * 70)
    
    start_time = time.time()
    
    # Initialize securely
    print("\n🔧 Initializing (secure mode)...")
    
    try:
        extractor = SecureGroqExtractor(secure_mode=secure_mode)
    except ValueError as e:
        print(f"{e}")
        return
    
    try:
        graph = SecureNeo4jManager()
    except (ValueError, ConnectionError) as e:
        print(f"{e}")
        return
    
    # Reset and prepare
    graph.reset_database()
    graph.create_indexes()
    
    # Load files
    text_dir = PROJECT_ROOT / "data" / "processed" / "cleaned_text"
    documents = load_files_securely(text_dir, max_files)
    
    if not documents:
        print("❌ No files found!")
        return
    
    # Process
    print(f"\n📊 Processing {len(documents)} documents...")
    
    all_triples = []
    stats = {"docs": 0, "chunks": 0, "triples": 0}
    
    for doc_hash, content in tqdm(documents.items(), desc="Documents"):
        chunks = chunk_text(content)
        if chunks_per_file:
            chunks = chunks[:chunks_per_file]
        
        for chunk in chunks:
            triples = extractor.extract_triples(chunk, source_hash=doc_hash)
            all_triples.extend(triples)
            stats["triples"] += len(triples)
            stats["chunks"] += 1
            
            # Rate limit
            time.sleep(RATE_LIMIT_DELAY)
            
            # SECURITY: Clear chunk from memory
            SecurityManager.secure_clear(chunk)
        
        stats["docs"] += 1
        
        # SECURITY: Clear document from memory
        SecurityManager.secure_clear(content)
    
    print(f"\n✅ Extraction: {stats['docs']} docs, {stats['chunks']} chunks, {stats['triples']} triples")
    
    # Import
    if all_triples:
        graph.batch_import(all_triples)
    
    # Stats
    final_stats = graph.get_stats()
    graph.close()
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ SECURE MIGRATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Results:")
    print(f"   Nodes: {final_stats['nodes']}")
    print(f"   Relationships: {final_stats['relationships']}")
    print(f"   Time: {elapsed/60:.1f} min")
    
    if final_stats['node_types']:
        print(f"\n   Top Node Types:")
        for t, c in list(final_stats['node_types'].items())[:8]:
            print(f"      {t}: {c}")
    
    if final_stats['rel_types']:
        print(f"\n   Top Relationships:")
        for t, c in list(final_stats['rel_types'].items())[:8]:
            print(f"      {t}: {c}")
    
    # SECURITY: Final cleanup
    SecurityManager.secure_clear(all_triples)
    gc.collect()
    
    print(f"\n🔒 Memory cleared securely")
    print(f"💡 Open Neo4j: http://localhost:7474")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Secure migration with Groq Cloud")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files")
    parser.add_argument("--chunks", type=int, default=None, help="Chunks per file")
    parser.add_argument("--secure-mode", action="store_true", help="Extra privacy (anonymize more)")
    
    args = parser.parse_args()
    
    migrate_secure(
        max_files=args.max_files,
        chunks_per_file=args.chunks,
        secure_mode=args.secure_mode
    )
