"""FastAPI backend for the Dangote Cement RAG system."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import json
import pandas as pd

from retriever import DangoteCementRetriever
from chains import DangoteCementRAGChains
from data_ingestion import DangoteCementDataIngestion
from config import STATIC_DIR, API_HOST, API_PORT, KEY_METRICS, ANALYSIS_YEARS
from utils import setup_logging

# Setup logging
logger = setup_logging(Path("logs/api.log"))

# Initialize FastAPI app
app = FastAPI(
    title="Dangote Cement RAG API",
    description="API for analyzing Dangote Cement annual reports using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global variables for components
retriever = None
chains = None
ingestion = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    query_type: Optional[str] = "auto"

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query_type: str

class MetricRequest(BaseModel):
    metric: str
    years: Optional[List[int]] = None
    segment: Optional[str] = None

class ChartDataResponse(BaseModel):
    labels: List[str]
    datasets: List[Dict[str, Any]]

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global retriever, chains, ingestion
    
    try:
        logger.info("Initializing RAG components...")
        
        # Initialize data ingestion
        ingestion = DangoteCementDataIngestion()
        
        # Initialize retriever
        retriever = DangoteCementRetriever(
            embedding_type="sentence_transformers",
            vectorstore_type="chroma"
        )
        
        # Initialize chains
        chains = DangoteCementRAGChains(retriever)
        
        logger.info("RAG components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        # Continue without components for basic functionality

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    else:
        return HTMLResponse(
            content="<h1>Dangote Cement RAG System</h1><p>Please create static/index.html</p>",
            status_code=200
        )

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system."""
    if not chains:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        result = chains.run(request.query, k=request.k)
        
        return QueryResponse(
            answer=result.get("answer", "No answer generated"),
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            query_type=result.get("query_type", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_available_metrics():
    """Get list of available financial metrics."""
    return {
        "metrics": KEY_METRICS,
        "years": ANALYSIS_YEARS
    }

@app.post("/api/metric-data")
async def get_metric_data(request: MetricRequest):
    """Get time series data for a specific metric."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    try:
        years = request.years or ANALYSIS_YEARS
        
        # Get time series data
        time_series = retriever.numeric_retriever.retrieve_time_series(
            request.metric, years
        )
        
        return {
            "metric": request.metric,
            "data": time_series.get("time_series", {}),
            "years": time_series.get("years", [])
        }
        
    except Exception as e:
        logger.error(f"Error getting metric data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chart-data/{metric}")
async def get_chart_data(
    metric: str,
    years: Optional[str] = Query(None, description="Comma-separated years")
):
    """Get chart data for visualization."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Chart data not available")
    
    try:
        # Parse years parameter
        if years:
            year_list = [int(y.strip()) for y in years.split(",")]
        else:
            year_list = ANALYSIS_YEARS
        
        # Get time series data
        time_series = retriever.numeric_retriever.retrieve_time_series(metric, year_list)
        data = time_series.get("time_series", {})
        
        if not data:
            return ChartDataResponse(labels=[], datasets=[])
        
        # Format for Chart.js
        labels = [str(year) for year in sorted(data.keys())]
        values = [data[int(year)] for year in labels]
        
        # Determine chart color based on metric
        color_map = {
            "Revenue": "rgb(54, 162, 235)",
            "Net Profit": "rgb(75, 192, 192)",
            "EBITDA": "rgb(255, 205, 86)",
            "Total Assets": "rgb(255, 99, 132)",
            "Total Debt": "rgb(153, 102, 255)"
        }
        
        color = color_map.get(metric, "rgb(201, 203, 207)")
        
        dataset = {
            "label": metric,
            "data": values,
            "borderColor": color,
            "backgroundColor": color.replace("rgb", "rgba").replace(")", ", 0.2)"),
            "tension": 0.1
        }
        
        return ChartDataResponse(
            labels=labels,
            datasets=[dataset]
        )
        
    except Exception as e:
        logger.error(f"Error getting chart data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/segments/{metric}/{year}")
async def get_segment_comparison(metric: str, year: int):
    """Get segment comparison data."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    try:
        segments = ["Nigeria", "Pan-Africa", "Cement", "Sugar"]
        
        comparison = retriever.numeric_retriever.compare_segments(
            metric, year, segments
        )
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error getting segment comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def ingest_documents():
    """Trigger document ingestion process."""
    if not ingestion:
        raise HTTPException(status_code=503, detail="Ingestion not initialized")
    
    try:
        logger.info("Starting document ingestion...")
        
        results = ingestion.process_all_reports()
        
        # Add documents to retriever if available
        if retriever and results.get("unstructured_data"):
            from langchain.schema import Document
            
            documents = []
            for item in results["unstructured_data"]:
                doc = Document(
                    page_content=item["text"],
                    metadata=item["metadata"]
                )
                documents.append(doc)
            
            retriever.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to retriever")
        
        return {
            "status": "success",
            "processed_files": len(results.get("processed_files", [])),
            "structured_tables": len(results.get("structured_data", [])),
            "text_chunks": len(results.get("unstructured_data", [])),
            "errors": results.get("errors", [])
        }
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "retriever": retriever is not None,
            "chains": chains is not None,
            "ingestion": ingestion is not None
        }
    }

@app.get("/api/logs")
async def get_recent_logs(lines: int = Query(50, description="Number of recent log lines")):
    """Get recent query logs."""
    try:
        from config import LOG_FILE
        
        if not LOG_FILE.exists():
            return {"logs": []}
        
        logs = []
        with open(LOG_FILE, 'r') as f:
            for line in f.readlines()[-lines:]:
                try:
                    log_entry = json.loads(line.strip())
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
        
        return {"logs": logs}
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting Dangote Cement RAG API server...")
    print(f"API will be available at: http://{API_HOST}:{API_PORT}")
    print(f"Web UI will be available at: http://{API_HOST}:{API_PORT}/static/")
    
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
