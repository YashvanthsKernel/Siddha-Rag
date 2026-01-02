// Neo4j Schema for Siddha Medicine Knowledge Graph
// This script defines the structure of the medical knowledge graph

// ============================================================================
// CONSTRAINTS - Ensure entity uniqueness
// ============================================================================

CREATE CONSTRAINT herb_name IF NOT EXISTS 
FOR (h:Herb) REQUIRE h.name IS UNIQUE;

CREATE CONSTRAINT medicine_name IF NOT EXISTS 
FOR (m:Medicine) REQUIRE m.name IS UNIQUE;

CREATE CONSTRAINT disease_name IF NOT EXISTS 
FOR (d:Disease) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT symptom_name IF NOT EXISTS 
FOR (s:Symptom) REQUIRE s.name IS UNIQUE;

CREATE CONSTRAINT treatment_name IF NOT EXISTS 
FOR (t:Treatment) REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT ingredient_name IF NOT EXISTS 
FOR (i:Ingredient) REQUIRE i.name IS UNIQUE;

CREATE CONSTRAINT side_effect_name IF NOT EXISTS 
FOR (se:SideEffect) REQUIRE se.name IS UNIQUE;

CREATE CONSTRAINT body_part_name IF NOT EXISTS 
FOR (bp:BodyPart) REQUIRE bp.name IS UNIQUE;

CREATE CONSTRAINT document_id IF NOT EXISTS 
FOR (doc:Document) REQUIRE doc.id IS UNIQUE;

// ============================================================================
// INDEXES - Speed up lookups
// ============================================================================

// Text search indexes
CREATE INDEX herb_properties IF NOT EXISTS 
FOR (h:Herb) ON (h.properties);

CREATE INDEX disease_category IF NOT EXISTS 
FOR (d:Disease) ON (d.category);

CREATE INDEX medicine_type IF NOT EXISTS 
FOR (m:Medicine) ON (m.type);

CREATE INDEX symptom_description IF NOT EXISTS 
FOR (s:Symptom) ON (s.description);

CREATE INDEX document_filename IF NOT EXISTS 
FOR (doc:Document) ON (doc.filename);

// ============================================================================
// VECTOR INDEXES - Enable hybrid search
// ============================================================================

// Vector index for herbs
CREATE VECTOR INDEX herb_embeddings IF NOT EXISTS
FOR (h:Herb) ON (h.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};

// Vector index for diseases
CREATE VECTOR INDEX disease_embeddings IF NOT EXISTS
FOR (d:Disease) ON (d.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};

// Vector index for medicines
CREATE VECTOR INDEX medicine_embeddings IF NOT EXISTS
FOR (m:Medicine) ON (m.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};

// Vector index for symptoms
CREATE VECTOR INDEX symptom_embeddings IF NOT EXISTS
FOR (s:Symptom) ON (s.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};

// Vector index for documents
CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
FOR (doc:Document) ON (doc.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};

// ============================================================================
// EXAMPLE DATA STRUCTURE
// ============================================================================
// Below are comments showing the expected structure of nodes and relationships

/*
// Example Herb Node
(:Herb {
  name: "Neem",
  scientific_name: "Azadirachta indica",
  properties: "Antibacterial, antifungal, anti-inflammatory",
  embedding: [0.123, 0.456, ...],  // 768-dimensional vector
  created_at: datetime(),
  updated_at: datetime()
})

// Example Disease Node
(:Disease {
  name: "Skin Infection",
  category: "Dermatological",
  symptoms: "Redness, itching, inflammation",
  embedding: [0.234, 0.567, ...],
  severity: "moderate"
})

// Example Relationship
(:Herb {name: "Neem"})-[:TREATS {
  efficacy: "high",
  dosage: "3-5g daily",
  duration: "7-14 days",
  confidence: 0.95,
  source_document: "doc_123"
}]->(:Disease {name: "Skin Infection"})

// Example Multi-hop Query
// Find herbs that treat diseases affecting specific body parts
MATCH (herb:Herb)-[:TREATS]->(disease:Disease)-[:AFFECTS]->(body:BodyPart {name: "Lungs"})
RETURN herb.name, disease.name, body.name

// Example Contraindication Check
// Check if two herbs interact
MATCH (h1:Herb {name: "Turmeric"})-[r:INTERACTS_WITH]-(h2:Herb)
RETURN h2.name, r.severity, r.description
*/
