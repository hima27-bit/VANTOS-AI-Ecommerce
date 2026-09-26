# AI-Powered E-Commerce Product Discovery & Recommendation Engine

An AI-powered e-commerce product discovery and recommendation system built for the **VANTOS Technologies internship project**.

The system combines semantic search, vector embeddings, Qwen LLM-based intent understanding and re-ranking, Retrieval-Augmented Generation (RAG), and recommendation techniques to improve natural-language product discovery.

---

## Project Objective

The objective of this project is to build an intelligent product discovery system that can:

- Understand natural-language customer queries
- Extract customer intent and preferences
- Retrieve semantically relevant products
- Apply filtering and re-ranking
- Generate grounded product recommendations
- Provide upselling and cross-selling suggestions
- Apply basic personalization

The system follows a two-pass search architecture:

1. **Pass 1:** Semantic vector retrieval
2. **Pass 2:** Qwen-based intent analysis, filtering and re-ranking

---

## System Architecture

```text
Customer Query
      │
      ▼
Query Understanding
      │
      ▼
Qwen Intent Extraction
      │
      ▼
Embedding Generation
      │
      ▼
Qdrant Vector Database
      │
      ▼
Top-K Product Retrieval
      │
      ▼
Filtering + Qwen Re-ranking
      │
      ▼
RAG Product Context
      │
      ├──────────────────┐
      ▼                  ▼
Recommendations    Upsell / Cross-sell
      │
      ▼
React Frontend
Technology Stack
Backend
Python
FastAPI
Pandas
AI / Search
Qwen3
Sentence Transformers
BAAI/bge-small-en-v1.5
Qdrant
Retrieval-Augmented Generation (RAG)
Semantic Search
LLM-based Re-ranking
Frontend
React
Vite
JavaScript / JSX
Testing & Evaluation
Pytest
Precision@K
Recall@K
MRR
NDCG@K
Intent Accuracy
Groundedness
Hallucination Rate
Latency
Dataset

The project uses a synthetic product catalog containing 600 products across six categories:

Mobiles
Laptops
Electronics
Fashion
Shoes
Accessories

The dataset contains:

Product attributes
Product descriptions
Features
Tags
Use cases
Pricing
Ratings
Availability
Related products
Upsell relationships

Dataset file:

products.csv
Embedding Pipeline

Product embeddings are generated from the following product information:

Product name
Description
Category
Features
Tags
Use cases

Embedding model:

BAAI/bge-small-en-v1.5

The resulting embeddings are stored in Qdrant for semantic similarity search.

The embedding pipeline is implemented in:

embedding_pipeline.py
Vector Database

The project uses Qdrant as the vector database.

Configuration:

Collection: products
Distance metric: Cosine
Embedding dimension: 384

The local Qdrant storage directory is:

qdrant_data/

This directory is generated locally and is intentionally excluded from Git.

Qwen LLM Integration

Qwen is used for:

Customer intent extraction
Query understanding
Product filtering
Product re-ranking
Grounded product explanations

The Qwen model is hosted locally using a llama.cpp OpenAI-compatible server.

Models Evaluated

The project evaluated:

Qwen3-1.7B
Qwen3-4B
Qwen3-1.7B Results
Metric	Result
Intent Accuracy	94.44%
Groundedness Rate	100%
Hallucination Rate	0%
Average Latency	2208.03 ms
P95 Latency	3178.76 ms
Qwen3-4B Results
Metric	Result
Intent Accuracy	94.44%
Groundedness Rate	100%
Hallucination Rate	0%
Average Latency	2233.25 ms
P95 Latency	3295.31 ms

Memory measurement was not captured in the current benchmark environment.

The evaluated models produced comparable results on the current benchmark, while Qwen3-1.7B had slightly lower measured latency.

Two-Pass Search

The search pipeline follows two main stages.

Pass 1 — Semantic Retrieval

The customer query is converted into an embedding using the Sentence Transformer model.

The embedding is then searched against the Qdrant product collection.

The system retrieves the most semantically relevant products.

Pass 2 — Qwen Processing

Qwen analyzes the customer intent and applies:

Category filtering
Budget constraints
Availability constraints
Preference matching
Relevance-based re-ranking

The final result combines semantic retrieval with Qwen-based relevance scoring.

Intent Extraction

Customer queries are converted into structured intent.

Example:

{
  "category": "laptops",
  "intent": "purchase",
  "max_price": 50000,
  "preferences": [
    "lightweight",
    "software development"
  ]
}

The extracted intent can contain:

Category
Intent
Maximum price
Product preferences
Constraints
RAG Pipeline

The system uses Retrieval-Augmented Generation to ground AI responses in available product information.

Before generating product-related responses, the system retrieves product context from the catalog.

Retrieved context may include:

Product name
Category
Description
Features
Use cases
Price
Rating
Specifications

This helps reduce unsupported product claims and keeps generated explanations grounded in available catalog information.

Recommendation Engine

The system provides several recommendation capabilities.

Primary Recommendations

Products are recommended based on:

Customer query
Extracted intent
Product attributes
Semantic relevance
Price
Ratings
Availability
Upselling

The system identifies higher-value alternatives and provides explanations based on additional product attributes and available catalog information.

Cross-selling

The system recommends complementary products such as:

Laptop accessories
Mouse
Keyboard
SSD
Bags
Chargers
Covers
Other related accessories
Personalization

Basic personalization adjusts recommendations using sample customer information such as:

Search history
Purchase history
Browsing context
Preferences
Preferred categories
Price preferences
Backend API

The backend is implemented using FastAPI.

Core Endpoints
Method	Endpoint	Purpose
GET	/api/health	Service health check
POST	/api/search	Natural-language product search
POST	/api/intent	Extract customer intent
POST	/api/recommendations	Generate recommendations
POST	/api/upsell	Generate upsell suggestions
POST	/api/cross-sell	Generate complementary products
POST	/api/embedding	Generate/test embeddings
GET	/api/products/{id}	Retrieve product details

The API also exposes FastAPI's automatically generated OpenAPI documentation.

Swagger UI:

http://127.0.0.1:8000/docs
Frontend

The frontend is built using React and Vite.

The interface provides:

Natural-language search
AI-powered product discovery
Category browsing
Product cards
Product details
AI search understanding
Recommendations
Upselling
Cross-selling
Wishlist interaction
Responsive layout

Frontend directory:

frontend/
Project Structure
VANTOS/
│
├── backend/
│   └── main.py
│
├── services/
│   └── search_service.py
│
├── evaluation/
│   ├── evaluate_ai.py
│   ├── evaluate_ai_task6.py
│   ├── evaluate_human_ground_truth.py
│   ├── evaluate_search.py
│   ├── evaluate_search_human_reviewed.py
│   ├── prepare_ground_truth.py
│   ├── results/
│   └── test/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── models/
│   └── Local Qwen models
│
├── products.csv
├── embedding_pipeline.py
├── start_vantos.ps1
├── .env.example
├── .gitignore
└── README.md
Files intentionally excluded from Git

The following are local/generated resources:

models/*.gguf
qdrant_data/
venv/
node_modules/
__pycache__/
.pytest_cache/
.env
Installation & Setup
Prerequisites

Recommended environment:

Windows
Python 3.10+
Node.js
npm
CUDA-capable GPU for faster local Qwen inference
Git
1. Clone the Repository
git clone <repository-url>
cd VANTOS
2. Create a Python Virtual Environment
python -m venv venv

Activate it on Windows:

.\venv\Scripts\Activate.ps1
3. Install Python Dependencies
pip install -r requirements.txt
4. Install Frontend Dependencies
cd frontend
npm install
cd ..
5. Configure Environment Variables

Copy:

.env.example

to:

.env

The .env file is intentionally excluded from Git.

6. Download a Qwen GGUF Model

Place the local Qwen model inside:

models/

For example:

models/
└── Qwen3-1.7B-Q4_K_M.gguf

The model files are intentionally excluded from Git because of their large size.

7. Start the Qwen Server

Using llama.cpp:

.\llama-server.exe `
  -m "C:\path\to\VANTOS\models\Qwen3-1.7B-Q4_K_M.gguf" `
  -ngl 99 `
  -c 8192 `
  --host 127.0.0.1 `
  --port 8080

Check the server:

http://127.0.0.1:8080/health

Expected response:

{
  "status": "ok"
}
8. Start the Backend

From the project root:

uvicorn backend.main:app --reload

Backend:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs
9. Start the Frontend

Open another terminal:

cd frontend
npm run dev

Vite will display the local frontend URL in the terminal.

Running the Embedding Pipeline

The embedding pipeline is implemented in:

embedding_pipeline.py

Run:

python embedding_pipeline.py

This generates embeddings for the product dataset and indexes them into the Qdrant collection.

Testing

The project contains unit and API tests under:

evaluation/test/

Run the complete test suite from the project root:

pytest

The tests cover:

Search service behavior
Intent extraction
Product retrieval
API endpoints
Recommendation-related functionality
Error handling
Evaluation

Evaluation scripts are available under:

evaluation/
AI Evaluation
python evaluation/evaluate_ai.py
Qwen Model Evaluation
python evaluation/evaluate_ai_task6.py
Search Evaluation
python evaluation/evaluate_search.py
Human-Reviewed Search Evaluation
python evaluation/evaluate_search_human_reviewed.py

Evaluation outputs are stored under:

evaluation/results/
Search Evaluation Metrics

The project evaluates:

Precision@K
Recall@K
Mean Reciprocal Rank (MRR)
NDCG@K
Search latency

The evaluated search approaches include:

Keyword search
Semantic search
Semantic search with Qwen re-ranking

The comparison uses the same evaluated query set and relevance judgments.

AI Evaluation Metrics

The AI evaluation includes:

Intent accuracy
Field-level intent accuracy
Groundedness
Hallucination rate
Latency
Evaluation Findings

The evaluation was designed to compare traditional keyword retrieval with semantic retrieval and the two-pass semantic + Qwen approach.

The project uses human-reviewed relevance judgments for the primary search comparison.

The current benchmark results are stored in:

evaluation/results/

The evaluation should be interpreted as a proof-of-concept benchmark on the project's dataset and reviewed query set rather than as a universal claim about e-commerce search performance.

Security & Repository Hygiene

The repository is configured to avoid committing sensitive or unnecessarily large files.

Excluded resources include:

Environment files
API keys
Passwords
Access tokens
Local Qwen model files
Virtual environments
Node modules
Qdrant local storage
Python cache files
Build artifacts
Log files

Use:

.env.example

as the configuration template.

Internship Requirement Coverage

The implementation covers the core internship requirements:

Requirement	Status
Business & technical research	Completed
Product dataset	Completed
Embedding pipeline	Completed
Vector database	Completed
Semantic search	Completed
Qwen integration	Completed
Two-pass search	Completed
Intent extraction	Completed
Recommendation engine	Completed
Upselling	Completed
Cross-selling	Completed
Basic personalization	Completed
RAG	Completed
FastAPI backend	Completed
React frontend	Completed
Unit/API testing	Completed
Search & AI evaluation	Completed
Traditional vs AI comparison	Completed

Advanced internship extensions such as a conversational AI shopping assistant and multilingual search are treated separately from the core implementation.

Limitations

The current implementation is a proof-of-concept system and has several limitations:

The product catalog is a synthetic dataset.
Qdrant is configured for local development.
Qwen inference is performed locally.
Search evaluation is based on a reviewed query set rather than production-scale traffic.
Business metrics such as conversion rate and add-to-cart influence are not measured.
Memory usage was not captured in the current Qwen benchmark.
The system does not represent a production payment, inventory, authentication, or order-management system.
Future Improvements

Possible future extensions include:

Conversational AI shopping assistant
Multilingual search
Voice-based product search
Larger product catalogs
Hybrid keyword + semantic retrieval
Advanced personalization
Production PostgreSQL integration
Cloud deployment
GPU inference optimization
User behavior analytics
Online recommendation evaluation
Dynamic product bundling
Project Status

The core internship proof-of-concept is implemented and evaluated.

The project demonstrates an end-to-end pipeline:

Natural Language Query
        ↓
Intent Understanding
        ↓
Embedding Generation
        ↓
Vector Retrieval
        ↓
Filtering
        ↓
Qwen Re-ranking
        ↓
RAG Context
        ↓
Recommendations
        ↓
React Web Interface
Internship Project

VANTOS Technologies Private Limited

Project Title

AI-Powered E-Commerce Product Discovery and Recommendation Engine using Qwen LLM, RAG and Semantic Search

Project Domain

Artificial Intelligence / Generative AI / E-Commerce

Primary Architecture

Semantic Retrieval + RAG + Qwen Re-ranking