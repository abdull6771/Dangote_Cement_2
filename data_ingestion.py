"""Data ingestion module for processing Dangote Cement annual reports."""

import pdfplumber
import pandas as pd
import tabula
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from config import PDF_DIR, CSV_DIR, CHUNKS_DIR, ANALYSIS_YEARS
from utils import extract_year_from_filename, create_metadata, setup_logging

logger = setup_logging(Path("logs/data_ingestion.log"))

class DangoteCementDataIngestion:
    """Main class for ingesting Dangote Cement annual report data."""
    
    def __init__(self, pdf_directory: Path = PDF_DIR):
        self.pdf_directory = pdf_directory
        self.csv_directory = CSV_DIR
        self.chunks_directory = CHUNKS_DIR
        
        # Ensure directories exist
        for directory in [self.csv_directory, self.chunks_directory]:
            directory.mkdir(exist_ok=True)
    
    def process_all_reports(self) -> Dict[str, Any]:
        """Process all PDF reports in the directory."""
        results = {
            "processed_files": [],
            "structured_data": [],
            "unstructured_data": [],
            "errors": []
        }
        
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                year = extract_year_from_filename(pdf_file.name)
                if not year or year not in ANALYSIS_YEARS:
                    logger.warning(f"Skipping {pdf_file.name} - year not in analysis range")
                    continue
                
                logger.info(f"Processing {pdf_file.name} for year {year}")
                
                # Extract structured data
                structured_data = self.extract_structured_data(pdf_file, year)
                results["structured_data"].extend(structured_data)
                
                # Extract unstructured data
                unstructured_data = self.extract_unstructured_data(pdf_file, year)
                results["unstructured_data"].extend(unstructured_data)
                
                results["processed_files"].append({
                    "filename": pdf_file.name,
                    "year": year,
                    "structured_tables": len(structured_data),
                    "text_chunks": len(unstructured_data)
                })
                
            except Exception as e:
                error_msg = f"Error processing {pdf_file.name}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        return results
    
    def extract_structured_data(self, pdf_path: Path, year: int) -> List[Dict[str, Any]]:
        """Extract structured financial tables from PDF."""
        structured_data = []
        
        try:
            # Use tabula to extract tables
            tables = tabula.read_pdf(
                str(pdf_path), 
                pages='all', 
                multiple_tables=True,
                pandas_options={'header': 0}
            )
            
            for i, table in enumerate(tables):
                if table.empty:
                    continue
                
                # Identify table type based on content
                table_type = self._identify_table_type(table)
                if not table_type:
                    continue
                
                # Clean and process table
                cleaned_table = self._clean_financial_table(table, year)
                
                if not cleaned_table.empty:
                    # Save as CSV
                    csv_filename = f"{year}_{table_type.replace(' ', '_').lower()}_{i}.csv"
                    csv_path = self.csv_directory / csv_filename
                    cleaned_table.to_csv(csv_path, index=False)
                    
                    # Create metadata
                    metadata = create_metadata(
                        year=year,
                        section_type=table_type,
                        page=i+1,  # Approximate page number
                        table_id=f"table_{i}"
                    )
                    
                    structured_data.append({
                        "data": cleaned_table,
                        "metadata": metadata,
                        "csv_path": str(csv_path)
                    })
                    
                    logger.info(f"Extracted {table_type} table with {len(cleaned_table)} rows")
        
        except Exception as e:
            logger.error(f"Error extracting structured data from {pdf_path}: {str(e)}")
        
        return structured_data
    
    def extract_unstructured_data(self, pdf_path: Path, year: int) -> List[Dict[str, Any]]:
        """Extract unstructured text from PDF."""
        unstructured_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # Identify sections
                    sections = self._identify_text_sections(text)
                    
                    for section_type, section_text in sections.items():
                        if not section_text.strip():
                            continue
                        
                        # Split into chunks
                        chunks = self._split_text_into_chunks(section_text)
                        
                        for chunk_id, chunk in enumerate(chunks):
                            metadata = create_metadata(
                                year=year,
                                section_type=section_type,
                                page=page_num + 1,
                                chunk_id=f"chunk_{chunk_id}"
                            )
                            
                            unstructured_data.append({
                                "text": chunk,
                                "metadata": metadata
                            })
                    
                    logger.debug(f"Processed page {page_num + 1} with {len(sections)} sections")
        
        except Exception as e:
            logger.error(f"Error extracting unstructured data from {pdf_path}: {str(e)}")
        
        return unstructured_data
    
    def _identify_table_type(self, table: pd.DataFrame) -> Optional[str]:
        """Identify the type of financial table based on content."""
        if table.empty:
            return None
        
        # Convert table to string for pattern matching
        table_str = table.to_string().lower()
        
        # Define patterns for different table types
        patterns = {
            "Income Statement": [
                "revenue", "turnover", "sales", "gross profit", "operating profit", 
                "net profit", "profit before tax", "profit after tax"
            ],
            "Balance Sheet": [
                "assets", "liabilities", "equity", "current assets", "non-current assets",
                "shareholders", "retained earnings"
            ],
            "Cash Flow Statement": [
                "cash flow", "operating activities", "investing activities", 
                "financing activities", "net cash"
            ],
            "Segment Reporting": [
                "segment", "nigeria", "pan-africa", "geographical", "business segment"
            ]
        }
        
        # Score each table type
        scores = {}
        for table_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in table_str)
            if score > 0:
                scores[table_type] = score
        
        # Return the table type with highest score
        if scores:
            return max(scores, key=scores.get)
        
        return "Other Financial Data"
    
    def _clean_financial_table(self, table: pd.DataFrame, year: int) -> pd.DataFrame:
        """Clean and standardize financial table."""
        if table.empty:
            return table
        
        # Add year column
        table = table.copy()
        table['Year'] = year
        
        # Clean column names
        table.columns = [str(col).strip() for col in table.columns]
        
        # Remove empty rows and columns
        table = table.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean numeric columns (will be handled by utils.clean_financial_number)
        return table
    
    def _identify_text_sections(self, text: str) -> Dict[str, str]:
        """Identify different sections in the text."""
        sections = {}
        
        # Define section patterns
        section_patterns = {
            "Chairman's Statement": [
                r"chairman['\s]*s?\s+statement",
                r"chairman['\s]*s?\s+message",
                r"chairman['\s]*s?\s+report"
            ],
            "CEO's Statement": [
                r"ceo['\s]*s?\s+statement",
                r"chief executive['\s]*s?\s+statement",
                r"managing director['\s]*s?\s+statement"
            ],
            "Risk Factors": [
                r"risk\s+factors?",
                r"principal\s+risks?",
                r"risk\s+management"
            ],
            "Sustainability": [
                r"sustainability",
                r"corporate\s+social\s+responsibility",
                r"csr\s+report"
            ],
            "ESG": [
                r"environmental[,\s]+social[,\s]+governance",
                r"esg\s+report",
                r"esg\s+performance"
            ],
            "Outlook": [
                r"outlook",
                r"future\s+prospects?",
                r"forward\s+looking"
            ]
        }
        
        text_lower = text.lower()
        
        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                if matches:
                    # Extract text around the match (next 2000 characters)
                    start_pos = matches[0].start()
                    end_pos = min(start_pos + 2000, len(text))
                    section_text = text[start_pos:end_pos]
                    sections[section_name] = section_text
                    break
        
        # If no specific sections found, treat as general business review
        if not sections:
            sections["Business Review"] = text[:2000]  # First 2000 characters
        
        return sections
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 800, 
                               overlap: int = 80) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks

def main():
    """Main function to run data ingestion."""
    ingestion = DangoteCementDataIngestion()
    results = ingestion.process_all_reports()
    
    print(f"Processed {len(results['processed_files'])} files")
    print(f"Extracted {len(results['structured_data'])} structured tables")
    print(f"Extracted {len(results['unstructured_data'])} text chunks")
    
    if results['errors']:
        print(f"Encountered {len(results['errors'])} errors:")
        for error in results['errors']:
            print(f"  - {error}")

if __name__ == "__main__":
    main()
