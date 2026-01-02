"""Test the improved RAG retrieval system."""
import sys
sys.path.insert(0, 'src')

from graph.hybrid_retriever import HybridRetriever

print("="*60)
print("Testing Vector Retrieval with Improvements")
print("="*60)

retriever = HybridRetriever(
    chroma_path='data/vectordb',
    neo4j_password='test'  # Neo4j may fail, but vector will work
)

# Test query
query = "What herbs treat fever in Siddha medicine?"
print(f"\nQuery: {query}")
print("-"*40)

results = retriever.retrieve_vector(query, top_k=5)
docs = results.get('documents', [])
metas = results.get('metadatas', [])
dists = results.get('distances', [])

print(f"\nFound {len(docs)} results")

for i in range(min(3, len(docs))):
    print(f"\n--- Result {i+1} (distance: {dists[i]:.3f}) ---")
    print(f"Source: {metas[i].get('filename', 'unknown')}")
    print(f"{docs[i][:300]}...")

retriever.close()
print("\n" + "="*60)
print("TEST COMPLETE!")
print("="*60)
