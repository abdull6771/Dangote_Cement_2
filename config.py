"""Configuration settings for Dangote Cement RAG pipeline."""

import os
from pathlib import Path
from typing import Dict, Any

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"

# Create directories if they don't exist
for dir_path in [DATA_DIR, LOGS_DIR, STATIC_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data subdirectories
PDF_DIR = DATA_DIR / "pdfs"
CSV_DIR = DATA_DIR / "csv"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

for dir_path in [PDF_DIR, CSV_DIR, CHUNKS_DIR, VECTOR_DB_DIR]:
    dir_path.mkdir(exist_ok=True)

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Embedding Configuration
EMBEDDING_MODELS = {
    "openai": "text-embedding-3-large",
    "bge": "BAAI/bge-large-en-v1.5",
    "sentence_transformers": "all-MiniLM-L6-v2"
}

# Vector Store Configuration
VECTOR_STORES = {
    "chroma": {"persist_directory": str(VECTOR_DB_DIR / "chroma")},
    "faiss": {"index_path": str(VECTOR_DB_DIR / "faiss_index")},
}

# Text Processing Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
MAX_TOKENS_PER_CHUNK = 1000

# Financial Data Configuration
FINANCIAL_TABLES = [
    "Income Statement",
    "Balance Sheet", 
    "Cash Flow Statement",
    "Segment Reporting",
    "Notes to Financial Statements"
]

KEY_METRICS = [
    "Revenue", "Net Profit", "EBITDA", "Total Assets", 
    "Total Liabilities", "Total Debt", "Total Equity", 
    "Dividend per Share", "Earnings per Share"
]

NARRATIVE_SECTIONS = [
    "Chairman's Statement",
    "CEO's Statement", 
    "Risk Factors",
    "Sustainability",
    "ESG",
    "Outlook",
    "Business Review"
]

# Years to analyze
ANALYSIS_YEARS = list(range(2018, 2024))

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "query_log.txt"
