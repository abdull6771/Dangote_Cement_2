"""Retrieval module for the Dangote Cement RAG pipeline."""

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langchain.schema import Document
from embeddings_factory import EmbeddingFactory, BaseEmbedding
from vectorstore_factory import VectorStoreFactory, BaseVectorStore
from config import CSV_DIR, KEY_METRICS, ANALYSIS_YEARS
from utils import setup_logging, clean_financial_number

logger = setup_logging(Path("logs/retriever.log"))

class HybridRetriever:
    """Hybrid retriever combining dense and sparse retrieval."""
    
    def __init__(self, vectorstore: BaseVectorStore, embedding: BaseEmbedding):
        self.vectorstore = vectorstore
        self.embedding = embedding
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.documents = []
        
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to both dense and sparse indexes."""
        # Add to vector store (dense)
        self.vectorstore.add_documents(documents)
        
        # Add to sparse index (TF-IDF)
        self.documents.extend(documents)
        texts = [doc.page_content for doc in self.documents]
        
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info(f"Updated TF-IDF index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error updating TF-IDF index: {e}")
    
    def retrieve(self, query: str, k: int = 5, alpha: float = 0.7, 
                filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Retrieve documents using hybrid approach."""
        
        # Dense retrieval (semantic)
        dense_results = self._dense_retrieve(query, k * 2, filters)
        
        # Sparse retrieval (lexical)
        sparse_results = self._sparse_retrieve(query, k * 2)
        
        # Combine and re-rank results
        combined_results = self._combine_results(
            dense_results, sparse_results, alpha, k
        )
        
        return combined_results
    
    def _dense_retrieve(self, query: str, k: int, 
                       filters: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Dense retrieval using vector similarity."""
        try:
            if filters:
                # Apply metadata filters
                results = self.vectorstore.similarity_search_with_score(
                    query, k=k, filter=filters
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            return results
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            return []
    
    def _sparse_retrieve(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Sparse retrieval using TF-IDF."""
        if self.tfidf_matrix is None or not self.documents:
            return []
        
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include non-zero similarities
                    results.append((self.documents[idx], similarities[idx]))
            
            return results
        except Exception as e:
            logger.error(f"Error in sparse retrieval: {e}")
            return []
    
    def _combine_results(self, dense_results: List[Tuple[Document, float]], 
                        sparse_results: List[Tuple[Document, float]], 
                        alpha: float, k: int) -> List[Document]:
        """Combine and re-rank dense and sparse results."""
        
        # Normalize scores
        dense_scores = self._normalize_scores([score for _, score in dense_results])
        sparse_scores = self._normalize_scores([score for _, score in sparse_results])
        
        # Create combined score dictionary
        doc_scores = {}
        
        # Add dense results
        for i, (doc, _) in enumerate(dense_results):
            doc_key = doc.page_content[:100]  # Use first 100 chars as key
            doc_scores[doc_key] = {
                'document': doc,
                'dense_score': dense_scores[i] if i < len(dense_scores) else 0,
                'sparse_score': 0
            }
        
        # Add sparse results
        for i, (doc, _) in enumerate(sparse_results):
            doc_key = doc.page_content[:100]
            if doc_key in doc_scores:
                doc_scores[doc_key]['sparse_score'] = sparse_scores[i] if i < len(sparse_scores) else 0
            else:
                doc_scores[doc_key] = {
                    'document': doc,
                    'dense_score': 0,
                    'sparse_score': sparse_scores[i] if i < len(sparse_scores) else 0
                }
        
        # Calculate combined scores
        for doc_key in doc_scores:
            dense_score = doc_scores[doc_key]['dense_score']
            sparse_score = doc_scores[doc_key]['sparse_score']
            doc_scores[doc_key]['combined_score'] = alpha * dense_score + (1 - alpha) * sparse_score
        
        # Sort by combined score and return top-k
        sorted_docs = sorted(
            doc_scores.values(), 
            key=lambda x: x['combined_score'], 
            reverse=True
        )
        
        return [item['document'] for item in sorted_docs[:k]]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]

class NumericRetriever:
    """Specialized retriever for numeric/financial data."""
    
    def __init__(self, csv_directory: Path = CSV_DIR):
        self.csv_directory = csv_directory
        self.financial_data = self._load_financial_data()
        
    def _load_financial_data(self) -> pd.DataFrame:
        """Load all financial data from CSV files."""
        all_data = []
        
        csv_files = list(self.csv_directory.glob("*.csv"))
        logger.info(f"Loading {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Extract metadata from filename
                filename_parts = csv_file.stem.split('_')
                if len(filename_parts) >= 2:
                    year = int(filename_parts[0])
                    table_type = ' '.join(filename_parts[1:-1]).replace('_', ' ').title()
                    
                    df['Year'] = year
                    df['Table_Type'] = table_type
                    df['Source_File'] = csv_file.name
                    
                    all_data.append(df)
                    
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Loaded financial data: {len(combined_df)} rows")
            return combined_df
        else:
            logger.warning("No financial data loaded")
            return pd.DataFrame()
    
    def retrieve_metric(self, metric: str, year: Optional[int] = None, 
                       segment: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve specific financial metric."""
        if self.financial_data.empty:
            return {"error": "No financial data available"}
        
        # Filter data
        filtered_data = self.financial_data.copy()
        
        # Filter by metric (fuzzy matching)
        metric_pattern = re.compile(metric, re.IGNORECASE)
        metric_matches = filtered_data.apply(
            lambda row: any(metric_pattern.search(str(cell)) for cell in row), 
            axis=1
        )
        filtered_data = filtered_data[metric_matches]
        
        # Filter by year
        if year:
            filtered_data = filtered_data[filtered_data['Year'] == year]
        
        # Filter by segment
        if segment:
            segment_pattern = re.compile(segment, re.IGNORECASE)
            segment_matches = filtered_data.apply(
                lambda row: any(segment_pattern.search(str(cell)) for cell in row), 
                axis=1
            )
            filtered_data = filtered_data[segment_matches]
        
        if filtered_data.empty:
            return {"error": f"No data found for metric: {metric}"}
        
        # Extract numeric values
        results = []
        for _, row in filtered_data.iterrows():
            for col in row.index:
                if col not in ['Year', 'Table_Type', 'Source_File']:
                    value = clean_financial_number(str(row[col]))
                    if value is not None:
                        results.append({
                            'metric': metric,
                            'value': value,
                            'year': row['Year'],
                            'table_type': row['Table_Type'],
                            'column': col,
                            'source_file': row['Source_File']
                        })
        
        return {
            "results": results,
            "count": len(results)
        }
    
    def retrieve_time_series(self, metric: str, 
                           years: Optional[List[int]] = None) -> Dict[str, Any]:
        """Retrieve time series data for a metric."""
        if years is None:
            years = ANALYSIS_YEARS
        
        time_series = {}
        for year in years:
            result = self.retrieve_metric(metric, year)
            if "results" in result and result["results"]:
                # Take the first (most relevant) result for each year
                time_series[year] = result["results"][0]["value"]
        
        return {
            "metric": metric,
            "time_series": time_series,
            "years": list(time_series.keys())
        }
    
    def compare_segments(self, metric: str, year: int, 
                        segments: List[str]) -> Dict[str, Any]:
        """Compare metric across different segments."""
        comparison = {}
        
        for segment in segments:
            result = self.retrieve_metric(metric, year, segment)
            if "results" in result and result["results"]:
                comparison[segment] = result["results"][0]["value"]
        
        return {
            "metric": metric,
            "year": year,
            "comparison": comparison
        }

class DangoteCementRetriever:
    """Main retriever class combining hybrid and numeric retrieval."""
    
    def __init__(self, embedding_type: str = "sentence_transformers", 
                 vectorstore_type: str = "chroma"):
        
        # Initialize components
        self.embedding = EmbeddingFactory.create_embedding(embedding_type)
        self.vectorstore = VectorStoreFactory.create_vectorstore(
            vectorstore_type, self.embedding
        )
        
        self.hybrid_retriever = HybridRetriever(self.vectorstore, self.embedding)
        self.numeric_retriever = NumericRetriever()
        
        logger.info(f"Initialized retriever with {embedding_type} embeddings and {vectorstore_type} vector store")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the retrieval system."""
        self.hybrid_retriever.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to retrieval system")
    
    def retrieve(self, query: str, k: int = 5, 
                query_type: str = "auto") -> Dict[str, Any]:
        """Main retrieval method that routes queries appropriately."""
        
        # Determine query type if auto
        if query_type == "auto":
            query_type = self._classify_query(query)
        
        logger.info(f"Processing {query_type} query: {query}")
        
        if query_type == "numeric":
            return self._handle_numeric_query(query)
        elif query_type == "narrative":
            return self._handle_narrative_query(query, k)
        elif query_type == "comparative":
            return self._handle_comparative_query(query, k)
        else:
            # Default to narrative
            return self._handle_narrative_query(query, k)
    
    def _classify_query(self, query: str) -> str:
        """Classify query type based on content."""
        query_lower = query.lower()
        
        # Numeric query patterns
        numeric_patterns = [
            r'what\s+was\s+.*\s+(revenue|profit|ebitda|assets|debt|equity)',
            r'(revenue|profit|ebitda|assets|debt|equity)\s+in\s+\d{4}',
            r'how\s+much\s+.*\s+(revenue|profit|ebitda)',
            r'\d{4}\s+(revenue|profit|ebitda|assets)'
        ]
        
        for pattern in numeric_patterns:
            if re.search(pattern, query_lower):
                return "numeric"
        
        # Comparative query patterns
        comparative_patterns = [
            r'compare.*\s+(vs|versus|compared\s+to)',
            r'(difference|comparison)\s+between',
            r'how\s+did.*compare',
            r'\d{4}\s+(vs|versus)\s+\d{4}',
            r'trends?\s+(from|between|across)'
        ]
        
        for pattern in comparative_patterns:
            if re.search(pattern, query_lower):
                return "comparative"
        
        # Default to narrative
        return "narrative"
    
    def _handle_numeric_query(self, query: str) -> Dict[str, Any]:
        """Handle numeric/financial queries."""
        # Extract metric and year from query
        metric, year, segment = self._extract_query_components(query)
        
        if metric:
            result = self.numeric_retriever.retrieve_metric(metric, year, segment)
            
            # Add query classification
            result["query_type"] = "numeric"
            result["query"] = query
            
            return result
        else:
            return {
                "error": "Could not extract metric from query",
                "query_type": "numeric",
                "query": query
            }
    
    def _handle_narrative_query(self, query: str, k: int) -> Dict[str, Any]:
        """Handle narrative/text-based queries."""
        # Extract filters from query
        filters = self._extract_filters(query)
        
        # Retrieve relevant documents
        documents = self.hybrid_retriever.retrieve(query, k, filters=filters)
        
        return {
            "query_type": "narrative",
            "query": query,
            "documents": documents,
            "count": len(documents)
        }
    
    def _handle_comparative_query(self, query: str, k: int) -> Dict[str, Any]:
        """Handle comparative queries."""
        # Try to extract years or segments for comparison
        years = self._extract_years(query)
        segments = self._extract_segments(query)
        metric = self._extract_metric(query)
        
        results = {
            "query_type": "comparative",
            "query": query
        }
        
        # If we can identify a specific metric and years, use numeric comparison
        if metric and len(years) >= 2:
            time_series = self.numeric_retriever.retrieve_time_series(metric, years)
            results["time_series"] = time_series
        
        # If we can identify segments, compare across segments
        if metric and segments and len(segments) >= 2:
            year = years[0] if years else max(ANALYSIS_YEARS)
            comparison = self.numeric_retriever.compare_segments(metric, year, segments)
            results["segment_comparison"] = comparison
        
        # Always include narrative results for context
        documents = self.hybrid_retriever.retrieve(query, k)
        results["documents"] = documents
        results["count"] = len(documents)
        
        return results
    
    def _extract_query_components(self, query: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Extract metric, year, and segment from query."""
        metric = self._extract_metric(query)
        year = self._extract_year(query)
        segment = self._extract_segment(query)
        
        return metric, year, segment
    
    def _extract_metric(self, query: str) -> Optional[str]:
        """Extract financial metric from query."""
        query_lower = query.lower()
        
        for metric in KEY_METRICS:
            if metric.lower() in query_lower:
                return metric
        
        # Check for common variations
        metric_variations = {
            'sales': 'Revenue',
            'turnover': 'Revenue',
            'income': 'Net Profit',
            'earnings': 'Net Profit',
            'assets': 'Total Assets',
            'liabilities': 'Total Liabilities',
            'debt': 'Total Debt',
            'equity': 'Total Equity'
        }
        
        for variation, standard in metric_variations.items():
            if variation in query_lower:
                return standard
        
        return None
    
    def _extract_year(self, query: str) -> Optional[int]:
        """Extract year from query."""
        year_match = re.search(r'20\d{2}', query)
        if year_match:
            year = int(year_match.group())
            if year in ANALYSIS_YEARS:
                return year
        return None
    
    def _extract_years(self, query: str) -> List[int]:
        """Extract multiple years from query."""
        years = []
        year_matches = re.findall(r'20\d{2}', query)
        for year_str in year_matches:
            year = int(year_str)
            if year in ANALYSIS_YEARS:
                years.append(year)
        return sorted(list(set(years)))
    
    def _extract_segment(self, query: str) -> Optional[str]:
        """Extract business segment from query."""
        query_lower = query.lower()
        
        segments = ['nigeria', 'pan-africa', 'cement', 'sugar', 'salt', 'flour']
        
        for segment in segments:
            if segment in query_lower:
                return segment
        
        return None
    
    def _extract_segments(self, query: str) -> List[str]:
        """Extract multiple segments from query."""
        query_lower = query.lower()
        segments = []
        
        segment_keywords = ['nigeria', 'pan-africa', 'cement', 'sugar', 'salt', 'flour']
        
        for segment in segment_keywords:
            if segment in query_lower:
                segments.append(segment)
        
        return segments
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract metadata filters from query."""
        filters = {}
        
        # Extract year filter
        year = self._extract_year(query)
        if year:
            filters["year"] = year
        
        # Extract section filter
        query_lower = query.lower()
        sections = ['chairman', 'ceo', 'risk', 'sustainability', 'esg', 'outlook']
        
        for section in sections:
            if section in query_lower:
                filters["section_type"] = section
                break
        
        return filters if filters else None

def test_retriever():
    """Test function for the retriever."""
    try:
        retriever = DangoteCementRetriever()
        
        # Test queries
        test_queries = [
            "What was Dangote Cement's revenue in 2021?",
            "Compare Nigeria vs Pan-Africa operations",
            "What risks were mentioned in the chairman's statement?",
            "Revenue trends from 2019 to 2023"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            result = retriever.retrieve(query)
            print(f"Query type: {result.get('query_type', 'unknown')}")
            
            if 'results' in result:
                print(f"Found {len(result['results'])} numeric results")
            if 'documents' in result:
                print(f"Found {len(result['documents'])} text documents")
            
    except Exception as e:
        print(f"Error testing retriever: {e}")

if __name__ == "__main__":
    test_retriever()
