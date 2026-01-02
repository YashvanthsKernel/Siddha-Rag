"""
Entity and Relationship Extractor
Uses LLM to extract structured knowledge from unstructured text.
"""

import ollama
import json
from typing import List, Dict, Optional, Tuple
from .config import (
    ENTITY_TYPES,
    RELATIONSHIP_TYPES,
    MAX_TRIPLES_PER_CHUNK,
    MIN_CONFIDENCE_THRESHOLD
)


class KnowledgeExtractor:
    """
    Extracts entities and relationships from text using LLM.
    Converts unstructured medical text into structured (subject, predicate, object) triples.
    """
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Initialize the knowledge extractor.
        
        Args:
            model: Ollama model name for extraction
        """
        self.model = model
        self.extraction_prompt_template = self._create_extraction_prompt()
    
    def _create_extraction_prompt(self) -> str:
        """Create the system prompt for triple extraction"""
        
        entity_types_str = ", ".join(ENTITY_TYPES)
        relationship_types_str = ", ".join(RELATIONSHIP_TYPES)
        
        prompt = f"""You are an expert in Siddha medicine and knowledge graph construction. 

Your task is to extract medical knowledge from text as structured triples (subject-predicate-object).

**ENTITY TYPES (use exact names):**
{entity_types_str}

**RELATIONSHIP TYPES (use exact names):**
{relationship_types_str}

**RULES:**
1. Extract ONLY factual medical information
2. Use EXACT entity and relationship type names listed above
3. Each triple must have: subject, subject_type, predicate, object, object_type, confidence (0.0-1.0)
4. Output maximum {MAX_TRIPLES_PER_CHUNK} most important triples
5. Output ONLY valid JSON array, no explanations

**OUTPUT FORMAT:**
[
  {{
    "subject": "entity name",
    "subject_type": "Herb|Medicine|Disease|...",
    "predicate": "TREATS|CONTAINS|...",
    "object": "entity name",
    "object_type": "Disease|Symptom|...",
    "confidence": 0.0-1.0
  }}
]

**EXAMPLE:**
Text: "Neem is used to treat skin infections. It has antibacterial properties."
Output:
[
  {{
    "subject": "Neem",
    "subject_type": "Herb",
    "predicate": "TREATS",
    "object": "Skin Infections",
    "object_type": "Disease",
    "confidence": 0.9
  }}
]
"""
        return prompt
    
    def extract_triples(self, text: str, max_retries: int = 2) -> List[Dict]:
        """
        Extract knowledge triples from text.
        
        Args:
            text: Medical text to process
            max_retries: Number of retry attempts on LLM failures
            
        Returns:
            List of validated triple dictionaries
        """
        if not text or len(text.strip()) < 20:
            return []
        
        # Construct full prompt
        full_prompt = f"{self.extraction_prompt_template}\n\nText to analyze:\n{text}\n\nExtract triples (JSON only):"
        
        for attempt in range(max_retries + 1):
            try:
                # Call LLM
                response = ollama.generate(
                    model=self.model,
                    prompt=full_prompt,
                    format="json",
                    options={
                        "temperature": 0.3,  # Lower temperature for more consistent extraction
                        "top_p": 0.9
                    }
                )
                
                # Parse JSON response
                triples = json.loads(response['response'])
                
                # Handle single triple vs array
                if isinstance(triples, dict):
                    triples = [triples]
                elif not isinstance(triples, list):
                    continue
                
                # Validate and filter triples
                valid_triples = [
                    triple for triple in triples 
                    if self._validate_triple(triple)
                ]
                
                # Filter by confidence threshold
                high_confidence_triples = [
                    triple for triple in valid_triples
                    if triple.get('confidence', 0) >= MIN_CONFIDENCE_THRESHOLD
                ]
                
                return high_confidence_triples[:MAX_TRIPLES_PER_CHUNK]
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    print(f"⚠️ JSON parse error, retry {attempt + 1}/{max_retries}")
                    continue
                else:
                    print(f"❌ Failed to parse JSON after {max_retries} retries: {e}")
                    return []
            except Exception as e:
                print(f"❌ Extraction error: {e}")
                return []
        
        return []
    
    def _validate_triple(self, triple: Dict) -> bool:
        """
        Validate that a triple has required fields and valid types.
        
        Args:
            triple: Triple dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields exist
        required_fields = ['subject', 'subject_type', 'predicate', 'object', 'object_type']
        if not all(field in triple for field in required_fields):
            return False
        
        # Check all fields are non-empty strings
        for field in required_fields:
            if not isinstance(triple[field], str) or not triple[field].strip():
                return False
        
        # Validate entity types
        if triple['subject_type'] not in ENTITY_TYPES:
            return False
        if triple['object_type'] not in ENTITY_TYPES:
            return False
        
        # Validate relationship type
        if triple['predicate'] not in RELATIONSHIP_TYPES:
            return False
        
        # Validate confidence (if present)
        if 'confidence' in triple:
            confidence = triple['confidence']
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                triple['confidence'] = 0.7  # Default confidence
        else:
            triple['confidence'] = 0.7
        
        return True
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract just entities (without relationships) from text.
        
        Args:
            text: Text to process
            
        Returns:
            List of (entity_name, entity_type) tuples
        """
        triples = self.extract_triples(text)
        
        # Collect unique entities
        entities = set()
        for triple in triples:
            entities.add((triple['subject'], triple['subject_type']))
            entities.add((triple['object'], triple['object_type']))
        
        return list(entities)
    
    def batch_extract(self, texts: List[str], show_progress: bool = True) -> List[List[Dict]]:
        """
        Extract triples from multiple text chunks.
        
        Args:
            texts: List of text strings
            show_progress: Whether to print progress
            
        Returns:
            List of triple lists (one per input text)
        """
        all_triples = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and i % 10 == 0:
                print(f"📊 Processing chunk {i+1}/{total}...")
            
            triples = self.extract_triples(text)
            all_triples.append(triples)
        
        if show_progress:
            total_triples = sum(len(t) for t in all_triples)
            print(f"✅ Extracted {total_triples} triples from {total} chunks")
        
        return all_triples


def main():
    """Example usage and testing"""
    
    # Initialize extractor
    extractor = KnowledgeExtractor()
    
    # Test text
    test_text = """
    Neem (Azadirachta indica) is widely used in Siddha medicine for treating skin diseases. 
    It has antibacterial and antifungal properties. Neem leaves can treat eczema and psoriasis.
    However, excessive use may cause liver toxicity. Neem should not be used by pregnant women.
    """
    
    print("🔍 Extracting knowledge from test text...\n")
    print(f"Text: {test_text}\n")
    
    # Extract triples
    triples = extractor.extract_triples(test_text)
    
    print(f"✅ Extracted {len(triples)} triples:\n")
    for i, triple in enumerate(triples, 1):
        print(f"{i}. ({triple['subject']})-[{triple['predicate']}]->({triple['object']})")
        print(f"   Types: {triple['subject_type']} → {triple['object_type']}")
        print(f"   Confidence: {triple['confidence']:.2f}\n")
    
    # Extract just entities
    entities = extractor.extract_entities(test_text)
    print(f"\n📝 Extracted {len(entities)} unique entities:")
    for entity_name, entity_type in entities:
        print(f"   • {entity_name} ({entity_type})")


if __name__ == "__main__":
    main()
