"""Utility functions for the Dangote Cement RAG pipeline."""

import re
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def clean_financial_number(value: str) -> Optional[float]:
    """Clean and convert financial numbers from text to float."""
    if not value or pd.isna(value):
        return None
    
    # Remove common formatting
    cleaned = str(value).strip()
    cleaned = re.sub(r'[,\s]', '', cleaned)
    cleaned = re.sub(r'[()()]', '-', cleaned)  # Parentheses indicate negative
    
    # Handle different units (millions, billions, thousands)
    multiplier = 1
    if 'billion' in cleaned.lower() or 'bn' in cleaned.lower():
        multiplier = 1_000_000_000
        cleaned = re.sub(r'(billion|bn)', '', cleaned, flags=re.IGNORECASE)
    elif 'million' in cleaned.lower() or 'mn' in cleaned.lower():
        multiplier = 1_000_000
        cleaned = re.sub(r'(million|mn)', '', cleaned, flags=re.IGNORECASE)
    elif 'thousand' in cleaned.lower() or 'k' in cleaned.lower():
        multiplier = 1_000
        cleaned = re.sub(r'(thousand|k)', '', cleaned, flags=re.IGNORECASE)
    
    # Extract numeric value
    try:
        # Find all numbers (including decimals)
        numbers = re.findall(r'-?\d+\.?\d*', cleaned)
        if numbers:
            return float(numbers[0]) * multiplier
    except (ValueError, IndexError):
        pass
    
    return None

def standardize_segment_name(segment: str) -> str:
    """Standardize segment names across years."""
    segment = segment.strip().lower()
    
    # Mapping for common variations
    segment_mapping = {
        'nigeria': 'Nigeria Operations',
        'nigeria operations': 'Nigeria Operations',
        'domestic': 'Nigeria Operations',
        'pan-africa': 'Pan-Africa Operations',
        'pan africa': 'Pan-Africa Operations',
        'rest of africa': 'Pan-Africa Operations',
        'other african countries': 'Pan-Africa Operations',
        'cement': 'Cement Operations',
        'sugar': 'Sugar Operations',
        'salt': 'Salt Operations',
        'flour': 'Flour Operations'
    }
    
    for key, value in segment_mapping.items():
        if key in segment:
            return value
    
    return segment.title()

def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from PDF filename."""
    # Look for 4-digit year pattern
    year_match = re.search(r'20\d{2}', filename)
    if year_match:
        return int(year_match.group())
    return None

def create_metadata(year: int, section_type: str, page: int, 
                   heading: str = "", chunk_id: str = "", 
                   table_id: str = "") -> Dict[str, Any]:
    """Create standardized metadata for documents."""
    return {
        "year": year,
        "section_type": section_type,
        "page": page,
        "heading": heading,
        "chunk_id": chunk_id,
        "table_id": table_id,
        "timestamp": datetime.now().isoformat()
    }

def save_query_log(query: str, sources: List[Dict], confidence: float, 
                  answer: str, log_file: Path):
    """Save query log with sources and confidence."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "sources": sources,
        "confidence": confidence,
        "answer": answer
    }
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def format_currency(amount: float, currency: str = "NGN") -> str:
    """Format currency amounts for display."""
    if amount >= 1_000_000_000:
        return f"{currency} {amount/1_000_000_000:.2f}B"
    elif amount >= 1_000_000:
        return f"{currency} {amount/1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"{currency} {amount/1_000:.2f}K"
    else:
        return f"{currency} {amount:.2f}"

def calculate_financial_ratios(data: Dict[str, float]) -> Dict[str, float]:
    """Calculate common financial ratios."""
    ratios = {}
    
    # Debt-to-Equity Ratio
    if data.get("Total Debt") and data.get("Total Equity"):
        ratios["debt_to_equity"] = data["Total Debt"] / data["Total Equity"]
    
    # Return on Assets (ROA)
    if data.get("Net Profit") and data.get("Total Assets"):
        ratios["roa"] = data["Net Profit"] / data["Total Assets"]
    
    # Return on Equity (ROE)
    if data.get("Net Profit") and data.get("Total Equity"):
        ratios["roe"] = data["Net Profit"] / data["Total Equity"]
    
    # EBITDA Margin
    if data.get("EBITDA") and data.get("Revenue"):
        ratios["ebitda_margin"] = data["EBITDA"] / data["Revenue"]
    
    # Net Profit Margin
    if data.get("Net Profit") and data.get("Revenue"):
        ratios["net_profit_margin"] = data["Net Profit"] / data["Revenue"]
    
    return ratios

def validate_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean financial data DataFrame."""
    # Remove rows with all NaN values
    df = df.dropna(how='all')
    
    # Clean numeric columns
    numeric_columns = df.select_dtypes(include=['object']).columns
    for col in numeric_columns:
        if col not in ['Year', 'Metric', 'Unit', 'Source']:
            df[col] = df[col].apply(clean_financial_number)
    
    return df
