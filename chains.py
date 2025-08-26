"""RAG chains for different types of queries."""

from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import json

try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    OpenAI = None
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None

from langchain.schema import Document
from retriever import DangoteCementRetriever
from utils import setup_logging, save_query_log, format_currency, calculate_financial_ratios
from config import LOG_FILE

logger = setup_logging(Path("logs/chains.log"))

class BaseChain:
    """Base class for RAG chains."""
    
    def __init__(self, retriever: DangoteCementRetriever, llm_model: str = "gpt-3.5-turbo"):
        self.retriever = retriever
        self.llm_model = llm_model
        
        # Initialize LLM if available
        if ChatOpenAI is not None:
            try:
                self.llm = ChatOpenAI(model_name=llm_model, temperature=0.1)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI LLM: {e}")
                self.llm = None
        else:
            logger.warning("OpenAI not available, using template-based responses")
            self.llm = None
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for citation."""
        citations = []
        for i, source in enumerate(sources, 1):
            if isinstance(source, Document):
                metadata = source.metadata
                citation = f"[{i}] Year: {metadata.get('year', 'N/A')}, "
                citation += f"Section: {metadata.get('section_type', 'N/A')}, "
                citation += f"Page: {metadata.get('page', 'N/A')}"
            elif isinstance(source, dict):
                citation = f"[{i}] Year: {source.get('year', 'N/A')}, "
                citation += f"Table: {source.get('table_type', 'N/A')}, "
                citation += f"Source: {source.get('source_file', 'N/A')}"
            else:
                citation = f"[{i}] {str(source)}"
            
            citations.append(citation)
        
        return "\n".join(citations)
    
    def _calculate_confidence(self, sources: List[Any], query: str) -> float:
        """Calculate confidence score for the answer."""
        if not sources:
            return 0.0
        
        # Simple confidence calculation based on number of sources and relevance
        base_confidence = min(len(sources) / 5.0, 1.0)  # Max confidence with 5+ sources
        
        # Boost confidence for numeric queries with exact matches
        if any(isinstance(s, dict) and 'value' in s for s in sources):
            base_confidence = min(base_confidence + 0.2, 1.0)
        
        return round(base_confidence, 2)

class NumericQAChain(BaseChain):
    """Chain for handling numeric/financial queries."""
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run numeric QA chain."""
        logger.info(f"Running numeric QA chain for: {query}")
        
        # Retrieve numeric data
        retrieval_result = self.retriever.retrieve(query, query_type="numeric")
        
        if "error" in retrieval_result:
            return {
                "answer": f"I couldn't find the requested financial data. {retrieval_result['error']}",
                "sources": [],
                "confidence": 0.0,
                "query_type": "numeric"
            }
        
        results = retrieval_result.get("results", [])
        if not results:
            return {
                "answer": "No financial data found for your query.",
                "sources": [],
                "confidence": 0.0,
                "query_type": "numeric"
            }
        
        # Format the answer
        answer = self._format_numeric_answer(query, results)
        confidence = self._calculate_confidence(results, query)
        
        # Log the query
        save_query_log(query, results, confidence, answer, LOG_FILE)
        
        return {
            "answer": answer,
            "sources": results,
            "confidence": confidence,
            "query_type": "numeric",
            "data": results
        }
    
    def _format_numeric_answer(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Format numeric answer with proper units and context."""
        if not results:
            return "No data found."
        
        # Group results by year
        by_year = {}
        for result in results:
            year = result.get('year')
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(result)
        
        answer_parts = []
        
        for year in sorted(by_year.keys()):
            year_results = by_year[year]
            
            if len(year_results) == 1:
                result = year_results[0]
                value = result['value']
                metric = result['metric']
                
                # Format currency if it's a financial metric
                if any(keyword in metric.lower() for keyword in ['revenue', 'profit', 'assets', 'debt', 'equity']):
                    formatted_value = format_currency(value)
                else:
                    formatted_value = f"{value:,.2f}"
                
                answer_parts.append(f"In {year}, {metric} was {formatted_value}")
            
            else:
                # Multiple results for the same year
                answer_parts.append(f"In {year}:")
                for result in year_results:
                    value = result['value']
                    metric = result['metric']
                    
                    if any(keyword in metric.lower() for keyword in ['revenue', 'profit', 'assets', 'debt', 'equity']):
                        formatted_value = format_currency(value)
                    else:
                        formatted_value = f"{value:,.2f}"
                    
                    answer_parts.append(f"  - {metric}: {formatted_value}")
        
        return "\n".join(answer_parts)

class NarrativeQAChain(BaseChain):
    """Chain for handling narrative/text-based queries."""
    
    def run(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Run narrative QA chain."""
        logger.info(f"Running narrative QA chain for: {query}")
        
        # Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(query, k=k, query_type="narrative")
        documents = retrieval_result.get("documents", [])
        
        if not documents:
            return {
                "answer": "I couldn't find relevant information to answer your query.",
                "sources": [],
                "confidence": 0.0,
                "query_type": "narrative"
            }
        
        # Generate answer using LLM or template
        if self.llm:
            answer = self._generate_llm_answer(query, documents)
        else:
            answer = self._generate_template_answer(query, documents)
        
        confidence = self._calculate_confidence(documents, query)
        
        # Log the query
        save_query_log(query, documents, confidence, answer, LOG_FILE)
        
        return {
            "answer": answer,
            "sources": documents,
            "confidence": confidence,
            "query_type": "narrative"
        }
    
    def _generate_llm_answer(self, query: str, documents: List[Document]) -> str:
        """Generate answer using LLM."""
        # Prepare context from documents
        context = "\n\n".join([
            f"Document {i+1} (Year: {doc.metadata.get('year', 'N/A')}, "
            f"Section: {doc.metadata.get('section_type', 'N/A')}):\n{doc.page_content}"
            for i, doc in enumerate(documents[:3])  # Use top 3 documents
        ])
        
        # Create prompt
        system_message = SystemMessage(content="""
        You are a financial analyst expert in Dangote Cement Plc. 
        Answer the user's question based on the provided context from annual reports.
        Be specific, accurate, and cite the relevant years and sections.
        If the information is not sufficient, say so clearly.
        """)
        
        human_message = HumanMessage(content=f"""
        Context from Dangote Cement annual reports:
        {context}
        
        Question: {query}
        
        Please provide a comprehensive answer based on the context above.
        """)
        
        try:
            response = self.llm([system_message, human_message])
            return response.content
        except Exception as e:
            logger.error(f"Error generating LLM answer: {e}")
            return self._generate_template_answer(query, documents)
    
    def _generate_template_answer(self, query: str, documents: List[Document]) -> str:
        """Generate answer using template-based approach."""
        answer_parts = ["Based on the available information from Dangote Cement annual reports:\n"]
        
        for i, doc in enumerate(documents[:3], 1):
            metadata = doc.metadata
            year = metadata.get('year', 'N/A')
            section = metadata.get('section_type', 'N/A')
            
            # Extract relevant snippet (first 200 characters)
            snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            answer_parts.append(f"{i}. From {year} {section}:")
            answer_parts.append(f"   {snippet}")
            answer_parts.append("")
        
        answer_parts.append("Please note: This is a summary based on available document excerpts. "
                          "For complete information, please refer to the full annual reports.")
        
        return "\n".join(answer_parts)

class ComparativeChain(BaseChain):
    """Chain for handling comparative queries."""
    
    def run(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Run comparative chain."""
        logger.info(f"Running comparative chain for: {query}")
        
        # Retrieve comparative data
        retrieval_result = self.retriever.retrieve(query, k=k, query_type="comparative")
        
        answer_parts = []
        sources = []
        
        # Handle time series comparison
        if "time_series" in retrieval_result:
            time_series = retrieval_result["time_series"]
            answer_parts.append(self._format_time_series_comparison(time_series))
            sources.extend([time_series])
        
        # Handle segment comparison
        if "segment_comparison" in retrieval_result:
            segment_comparison = retrieval_result["segment_comparison"]
            answer_parts.append(self._format_segment_comparison(segment_comparison))
            sources.extend([segment_comparison])
        
        # Handle narrative comparison
        documents = retrieval_result.get("documents", [])
        if documents:
            if self.llm:
                narrative_answer = self._generate_comparative_llm_answer(query, documents)
            else:
                narrative_answer = self._generate_comparative_template_answer(query, documents)
            
            answer_parts.append(narrative_answer)
            sources.extend(documents)
        
        # Combine all answer parts
        if answer_parts:
            answer = "\n\n".join(answer_parts)
        else:
            answer = "I couldn't find sufficient information to make the requested comparison."
        
        confidence = self._calculate_confidence(sources, query)
        
        # Log the query
        save_query_log(query, sources, confidence, answer, LOG_FILE)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "query_type": "comparative"
        }
    
    def _format_time_series_comparison(self, time_series: Dict[str, Any]) -> str:
        """Format time series comparison."""
        metric = time_series.get("metric", "Unknown Metric")
        data = time_series.get("time_series", {})
        
        if not data:
            return f"No time series data available for {metric}."
        
        answer_parts = [f"Time Series Analysis for {metric}:"]
        
        # Sort by year
        sorted_years = sorted(data.keys())
        
        for year in sorted_years:
            value = data[year]
            if isinstance(value, (int, float)):
                formatted_value = format_currency(value) if 'revenue' in metric.lower() or 'profit' in metric.lower() else f"{value:,.2f}"
                answer_parts.append(f"  {year}: {formatted_value}")
        
        # Calculate growth rates
        if len(sorted_years) >= 2:
            answer_parts.append("\nGrowth Analysis:")
            for i in range(1, len(sorted_years)):
                prev_year = sorted_years[i-1]
                curr_year = sorted_years[i]
                prev_value = data[prev_year]
                curr_value = data[curr_year]
                
                if prev_value and curr_value:
                    growth_rate = ((curr_value - prev_value) / prev_value) * 100
                    answer_parts.append(f"  {prev_year} to {curr_year}: {growth_rate:+.1f}%")
        
        return "\n".join(answer_parts)
    
    def _format_segment_comparison(self, segment_comparison: Dict[str, Any]) -> str:
        """Format segment comparison."""
        metric = segment_comparison.get("metric", "Unknown Metric")
        year = segment_comparison.get("year", "Unknown Year")
        data = segment_comparison.get("comparison", {})
        
        if not data:
            return f"No segment comparison data available for {metric} in {year}."
        
        answer_parts = [f"Segment Comparison for {metric} in {year}:"]
        
        # Sort by value (descending)
        sorted_segments = sorted(data.items(), key=lambda x: x[1] if x[1] else 0, reverse=True)
        
        total_value = sum(v for v in data.values() if v)
        
        for segment, value in sorted_segments:
            if value:
                formatted_value = format_currency(value) if 'revenue' in metric.lower() or 'profit' in metric.lower() else f"{value:,.2f}"
                percentage = (value / total_value) * 100 if total_value else 0
                answer_parts.append(f"  {segment}: {formatted_value} ({percentage:.1f}%)")
        
        return "\n".join(answer_parts)
    
    def _generate_comparative_llm_answer(self, query: str, documents: List[Document]) -> str:
        """Generate comparative answer using LLM."""
        # Group documents by year
        by_year = {}
        for doc in documents:
            year = doc.metadata.get('year', 'Unknown')
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(doc)
        
        # Prepare context
        context_parts = []
        for year in sorted(by_year.keys()):
            year_docs = by_year[year][:2]  # Max 2 docs per year
            context_parts.append(f"\n{year} Information:")
            for doc in year_docs:
                snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                context_parts.append(f"- {snippet}")
        
        context = "\n".join(context_parts)
        
        system_message = SystemMessage(content="""
        You are a financial analyst expert in Dangote Cement Plc.
        Compare and analyze the information across different years based on the provided context.
        Highlight key differences, trends, and changes over time.
        Be specific about years and provide quantitative comparisons where possible.
        """)
        
        human_message = HumanMessage(content=f"""
        Context from Dangote Cement annual reports across multiple years:
        {context}
        
        Comparative Question: {query}
        
        Please provide a detailed comparative analysis based on the context above.
        """)
        
        try:
            response = self.llm([system_message, human_message])
            return response.content
        except Exception as e:
            logger.error(f"Error generating comparative LLM answer: {e}")
            return self._generate_comparative_template_answer(query, documents)
    
    def _generate_comparative_template_answer(self, query: str, documents: List[Document]) -> str:
        """Generate comparative answer using template."""
        # Group by year
        by_year = {}
        for doc in documents:
            year = doc.metadata.get('year', 'Unknown')
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(doc)
        
        answer_parts = ["Comparative Analysis:"]
        
        for year in sorted(by_year.keys()):
            year_docs = by_year[year]
            answer_parts.append(f"\n{year}:")
            
            for doc in year_docs[:2]:  # Max 2 docs per year
                section = doc.metadata.get('section_type', 'Unknown Section')
                snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                answer_parts.append(f"  {section}: {snippet}")
        
        return "\n".join(answer_parts)

class DangoteCementRAGChains:
    """Main class combining all RAG chains."""
    
    def __init__(self, retriever: DangoteCementRetriever, llm_model: str = "gpt-3.5-turbo"):
        self.retriever = retriever
        
        # Initialize chains
        self.numeric_chain = NumericQAChain(retriever, llm_model)
        self.narrative_chain = NarrativeQAChain(retriever, llm_model)
        self.comparative_chain = ComparativeChain(retriever, llm_model)
        
        logger.info("Initialized all RAG chains")
    
    def run(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Run appropriate chain based on query type."""
        # Classify query type
        query_type = self.retriever._classify_query(query)
        
        if query_type == "numeric":
            return self.numeric_chain.run(query)
        elif query_type == "narrative":
            return self.narrative_chain.run(query, k)
        elif query_type == "comparative":
            return self.comparative_chain.run(query, k)
        else:
            # Default to narrative
            return self.narrative_chain.run(query, k)

def test_chains():
    """Test function for RAG chains."""
    try:
        from retriever import DangoteCementRetriever
        
        retriever = DangoteCementRetriever()
        chains = DangoteCementRAGChains(retriever)
        
        test_queries = [
            "What was Dangote Cement's revenue in 2021?",
            "What risks were mentioned in the 2020 annual report?",
            "Compare revenue growth from 2019 to 2023",
            "How did the chairman's outlook change between 2018 and 2022?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print('='*50)
            
            result = chains.run(query)
            
            print(f"Query Type: {result.get('query_type', 'unknown')}")
            print(f"Confidence: {result.get('confidence', 0.0)}")
            print(f"Answer: {result.get('answer', 'No answer')}")
            print(f"Sources: {len(result.get('sources', []))}")
            
    except Exception as e:
        print(f"Error testing chains: {e}")

if __name__ == "__main__":
    test_chains()
