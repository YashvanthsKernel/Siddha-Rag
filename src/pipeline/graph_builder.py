import json
import re

class GraphBuilder:
    def __init__(self):
        # In production, initialize your LLM here (e.g., LangChain + Ollama)
        pass

    def extract_relations(self, text):
        """
        Extracts entities and relations from text.
        Returns a list of dictionaries: {"head": "Herb", "type": "CURES", "tail": "Disease"}
        """
        relations = []
        
        # --- SIMPLE REGEX LOGIC (Replace this with LLM extraction) ---
        # Example pattern: "Neem cures skin infection"
        # This is just a placeholder to demonstrate the data structure.
        
        # 1. Look for 'cures' or 'treats'
        patterns = [
            r"(\w+)\s+(cures|treats|heals)\s+(\w+)",
            r"(\w+)\s+(is used for)\s+(\w+)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                source, relation, target = match
                relations.append({
                    "head": source,
                    "type": relation.upper().replace(" ", "_"),
                    "tail": target,
                    "meta": {"source_text": f"{source} {relation} {target}"}
                })
        
        return relations

if __name__ == "__main__":
    gb = GraphBuilder()
    sample = "Neem treats acne. Tulsi is used for cold."
    print(json.dumps(gb.extract_relations(sample), indent=2))
