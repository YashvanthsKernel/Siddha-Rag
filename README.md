<div align="center">

# 🌿 Siddha AI

### *Ancient Wisdom Meets Modern Intelligence*

**A Hybrid RAG System for Traditional Siddha Medicine Knowledge**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-4581C3?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai)

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Documentation](#-documentation)

---

</div>

## ✨ What is Siddha AI?

**Siddha AI** is an intelligent knowledge retrieval system that combines the ancient wisdom of **Siddha medicine** (one of the oldest medical systems from Tamil Nadu, India) with cutting-edge **AI technologies**.

> 🎯 **Ask natural language questions** about herbs, treatments, and remedies  
> 📚 **Get accurate answers** backed by authentic Siddha medicine texts  
> 🔗 **Explore relationships** between diseases, herbs, and treatments

---

## 🚀 Features

<table>
<tr>
<td width="50%">

### 🧠 Hybrid RAG Engine
Combines **vector search** (ChromaDB) with **knowledge graph** (Neo4j) for superior accuracy

### 🌐 Modern Web Interface  
Beautiful React + TypeScript frontend with dark mode support

### 🔒 Privacy-First
Runs entirely **locally** using Ollama - your data never leaves your machine

</td>
<td width="50%">

### 📊 Three Retrieval Modes
- **Vector**: Fast semantic search
- **Graph**: Entity relationships
- **Hybrid**: Best of both worlds ✨

### 📄 Multi-Format Support
Process **PDF**, **DOCX**, and **TXT** documents automatically

### ⚡ Real-Time Streaming
Watch responses generate in real-time with SSE

</td>
</tr>
</table>

---

## 🎬 Demo

<div align="center">

<!-- Add your demo GIF/screenshot here -->
| Ask a Question | Get Intelligent Answers |
|:-:|:-:|
| 🗣️ "What herbs treat fever in Siddha?" | 📖 Detailed response with sources |
| 🔍 Entity extraction & graph facts | 📚 Referenced documents |

*Coming soon: Demo video/screenshots*

</div>

---

## 📁 Project Structure

```
🌿 Siddha-LLM/
│
├── � api_server.py           # Flask REST API (Port 5001)
├── � requirements.txt        # Python dependencies
│
├── 🎨 app/                    # Frontend Application
│   ├── 📱 client/             # React + Vite
│   │   └── src/components/    # ChatInput, MessageBubble, etc.
│   └── 🔌 server/             # Express middleware
│       └── storage.ts         # Backend integration
│
├── 🧠 src/                    # Core Python Modules
│   ├── � rag/                # RAG System
│   │   ├── rag_system.py      # Main SiddhaRAG class
│   │   ├── retriever.py       # Document retrieval
│   │   ├── generator.py       # LLM response generation
│   │   └── embeddings.py      # Sentence transformers
│   │
│   ├── � pipeline/           # Document Processing
│   │   ├── main_pipeline.py   # Ingestion pipeline
│   │   └── chunking.py        # Text chunking
│   │
│   └── � graph/              # Knowledge Graph
│       ├── entity_extractor.py
│       ├── graph_builder.py
│       └── hybrid_retriever.py
│
├── �️ scripts/                # CLI Utilities
│   ├── setup_neo4j.py         # Initialize database
│   └── migrate_to_graph.py    # Populate graph
│
├── � data/                   # Data Storage
│   ├── raw/                   # Original documents
│   ├── processed/             # Cleaned texts
│   ├── vectordb/              # ChromaDB
│   └── graphdb/               # Neo4j exports
│
└── � documentation/          # Docs & Guides
```

---

## ⚡ Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| 🐍 Python | 3.11+ | Backend |
| 📦 Node.js | 18+ | Frontend |
| 🦙 Ollama | Latest | Local LLM |
| 🔗 Neo4j Desktop | 5.x | Graph (optional) |

### Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/YashvanthsKernel/Siddha-LLM.git
cd Siddha-LLM

# 2️⃣ Setup Python environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# 3️⃣ Setup Frontend
cd app && npm install && cd ..

# 4️⃣ Download Ollama model
ollama pull llama3.2:3b
```

### 🏃 Running the System

<table>
<tr>
<th>Terminal</th>
<th>Command</th>
<th>Service</th>
</tr>
<tr>
<td>1️⃣</td>
<td><code>ollama serve</code></td>
<td>🦙 LLM Engine</td>
</tr>
<tr>
<td>2️⃣</td>
<td><code>python api_server.py --password YOUR_PASS</code></td>
<td>🐍 Backend API</td>
</tr>
<tr>
<td>3️⃣</td>
<td><code>cd app && npm run dev</code></td>
<td>🎨 Frontend</td>
</tr>
</table>

### 🌐 Open in Browser

```
http://localhost:5000
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        🎨 Frontend (React)                       │
│                      http://localhost:5000                       │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼─────────────────────────────────────┐
│                    🐍 Flask API Server                           │
│                      Port 5001                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  /api/query │  │ /api/chats  │  │/api/suggest │              │
│  └──────┬──────┘  └─────────────┘  └─────────────┘              │
└─────────┼────────────────────────────────────────────────────────┘
          │
    ┌─────▼─────┐
    │ SiddhaRAG │
    │  System   │
    └─────┬─────┘
          │
    ┌─────┴─────────────────┐
    │                       │
┌───▼───┐              ┌────▼────┐
│ChromaDB│              │  Neo4j  │
│ Vector │              │  Graph  │
│  DB    │              │   DB    │
└───┬────┘              └────┬────┘
    │     ┌─────────┐        │
    └────►│ Ollama  │◄───────┘
          │  LLM    │
          └─────────┘
```

---

## � RAG Modes Comparison

| Mode | Speed | Accuracy | Best For |
|:----:|:-----:|:--------:|:---------|
| 🔍 **Vector** | ⚡⚡⚡ | ⭐⭐⭐ | General queries, fast responses |
| 🔗 **Graph** | ⚡⚡ | ⭐⭐⭐⭐ | Entity relationships, "treats what?" |
| 🌟 **Hybrid** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Best results (recommended) |

---

## 📖 API Reference

<details>
<summary><b>POST /api/query</b> - Query the RAG system</summary>

```json
{
  "question": "What herbs treat fever?",
  "strategy": "hybrid",
  "top_k": 5
}
```
</details>

<details>
<summary><b>GET /api/chats</b> - List chat sessions</summary>

Returns array of chat sessions with titles and timestamps
</details>

<details>
<summary><b>POST /api/query/stream</b> - Streaming response (SSE)</summary>

Real-time token streaming for chat interface
</details>

---

## 🛠️ Scripts

| Script | Description | Command |
|--------|-------------|---------|
| 🔧 `setup_neo4j.py` | Initialize Neo4j schema | `python scripts/setup_neo4j.py` |
| 📥 `migrate_to_graph.py` | Populate knowledge graph | `python scripts/migrate_to_graph.py` |
| 💾 `graph_export.py` | Backup graph to JSON | `python scripts/graph_export.py --export` |
| 📤 `import_graph_data.py` | Restore from backup | `python scripts/import_graph_data.py` |

---

## � Configuration

### Environment Variables

Create a `.env` file:

```env
NEO4J_PASSWORD=your_password_here
```

### Ports

| Service | Port |
|---------|------|
| Frontend | 5000 |
| Backend API | 5001 |
| Ollama | 11434 |
| Neo4j | 7687 |

---

## 📚 Documentation

Detailed documentation is available in the `documentation/` folder:

- 📖 [Project Journey](documentation/PROJECT_JOURNEY.md) - Development history
- 🔍 [Source Code Analysis](documentation/SOURCE_CODE_ANALYSIS.md)
- 🛠️ [Scripts Analysis](documentation/SCRIPTS_ANALYSIS.md)
- 🔄 [RAG Modes Guide](documentation/RAG_MODES.md)

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to branch (`git push origin feature/amazing-feature`)
5. 🎉 Open a Pull Request

---

## � License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

<table>
<tr>
<td align="center">
<img src="https://img.icons8.com/color/48/000000/herbal-medicine.png" width="40"/>
<br><b>Siddha Medicine</b>
<br>Ancient Tamil healing
</td>
<td align="center">
<img src="https://ollama.ai/public/ollama.png" width="40"/>
<br><b>Ollama</b>
<br>Local LLM inference
</td>
<td align="center">
<img src="https://www.trychroma.com/chroma-logo.png" width="40"/>
<br><b>ChromaDB</b>
<br>Vector database
</td>
<td align="center">
<img src="https://neo4j.com/favicon.ico" width="40"/>
<br><b>Neo4j</b>
<br>Graph database
</td>
</tr>
</table>

---

<div align="center">

### ⭐ Star this repo if you find it helpful!

Made with ❤️ for preserving traditional medicine knowledge

**[YashvanthsKernel](https://github.com/YashvanthsKernel)**

</div>
