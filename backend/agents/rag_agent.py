# backend/agents/rag_agent.py
import os
import logging
from typing import Dict, Any, List, Optional
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagAgent:
    def __init__(self):
        # Initialize Pinecone client
        api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "test1"
        
        # Change to OpenAI embeddings instead of HuggingFace
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=openai_api_key
        )
        self.llm = ChatOpenAI(temperature=0, api_key=openai_api_key)
        
        # Log available indexes for debugging
        logger.info(f"Available Pinecone indexes: {self.pc.list_indexes()}")
        
    def query(self, query_text: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Query the RAG system with optional metadata filtering by multiple years and quarters.
        
        Args:
            query_text: The query text
            years: Optional list of years to filter by
            quarters: Optional list of quarters to filter by
            
        Returns:
            Dictionary with retrieved context and generated response
        """
        try:
            # Log query parameters
            logger.info(f"RAG Query: '{query_text}', Years: {years}, Quarters: {quarters}")
            
            # Generate embedding for the query using OpenAI
            query_embedding = self.embedding_model.embed_query(query_text)
            
            # Prepare metadata filter - convert to format stored in Pinecone
            filter_dict = {}
            if years is not None and len(years) > 0:
                filter_dict["year"] = {"$in": [str(year) for year in years]}  # Filter for any of the years
            if quarters is not None and len(quarters) > 0:
                filter_dict["quarter"] = {"$in": [f"q{quarter}" for quarter in quarters]}  # Filter for any of the quarters
            
            logger.info(f"Using filter: {filter_dict}")
            
            # Connect to index
            index = self.pc.Index(self.index_name)
            
            # Perform hybrid search with metadata filtering
            search_results = index.query(
                vector=query_embedding,
                filter=filter_dict if filter_dict else None,
                top_k=10,
                include_metadata=True,
                alpha=0.5  # Hybrid search parameter - balance between metadata and vector similarity
            )
            
            # Log search results for debugging
            logger.info(f"Found {len(search_results.matches)} matches")
            
            # Extract retrieved contexts
            contexts = []
            for i, match in enumerate(search_results.matches):
                # Extract text and metadata
                text = match.metadata.get("text", "")
                source = match.metadata.get("source", "Unknown source")
                doc_year = match.metadata.get("year", "Unknown year")
                doc_quarter = match.metadata.get("quarter", "Unknown quarter")
                
                # Log match details for debugging
                logger.info(f"Match {i+1}: Score {match.score}, Source: {source}, Year: {doc_year}, Quarter: {doc_quarter}")
                
                # Add formatted context
                contexts.append(f"[Source: {source}, Year: {doc_year}, Quarter: {doc_quarter}]\n{text}")
            
            # If no contexts were retrieved, provide a fallback
            if not contexts:
                logger.warning("No relevant information found in Pinecone index")
                return {
                    "context": "",
                    "response": "I couldn't find any relevant information about NVIDIA based on your query and filters. Please try a different query or adjust the year/quarter filters.",
                    "sources": []
                }
            
            # Combine contexts for the LLM
            combined_context = "\n\n".join(contexts)
            
            # Create prompt for generation
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a financial research assistant specializing in NVIDIA.
                Use the following historical information from NVIDIA quarterly reports to answer the query.
                Only use the information provided in the context.
                If the information is not in the context, say so clearly.
                Be specific about any financial figures, percentages, or trends mentioned in the context.
                Format your response in a clear, concise manner suitable for financial analysis.
                """),
                ("human", "Context information:\n{context}\n\nQuery: {query}")
            ])
            
            # Generate response using LLM
            logger.info("Generating response with LLM")
            chain = prompt | self.llm
            response = chain.invoke({
                "context": combined_context,
                "query": query_text
            })
            
            logger.info("LLM response generated successfully")
            
            # Return results
            return {
                "context": combined_context,
                "response": response.content,
                "sources": [match.metadata.get("source", "Unknown") for match in search_results.matches]
            }
        except Exception as e:
            # Add better error handling
            logger.error(f"Error in RAG agent: {str(e)}", exc_info=True)
            return {
                "context": "",
                "response": f"Error retrieving information: {str(e)}",
                "sources": []
            }
    
    def test_connection(self) -> bool:
        """Test connection to Pinecone and verify index exists"""
        try:
            indexes = self.pc.list_indexes()
            logger.info(f"Available indexes: {indexes}")
            
            # Check if our index exists in the available indexes
            if self.index_name not in [idx for idx in indexes]:
                logger.error(f"Index '{self.index_name}' not found in available indexes: {indexes}")
                return False
                
            # Test a simple query to make sure we can connect
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
            return True
        except Exception as e:
            logger.error(f"Error testing Pinecone connection: {str(e)}", exc_info=True)
            return False