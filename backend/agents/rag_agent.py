# # backend/agents/rag_agent.py
# import os
# import logging
# from typing import Dict, Any, List, Optional
# from pinecone import Pinecone
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.prompts import ChatPromptTemplate

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class RagAgent:
#     def __init__(self):
#         # Directly read the .env file to get the API key
#         backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         env_file = os.path.join(backend_dir, '.env')
        
#         # Load environment variables directly from .env file
#         env_vars = {}
#         with open(env_file, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line or line.startswith('#'):
#                     continue
#                 if '=' in line:
#                     key, value = line.split('=', 1)
#                     env_vars[key.strip()] = value.strip()
        
#         # Get API keys from loaded variables
#         api_key = env_vars.get("PINECONE_API_KEY")
#         openai_api_key = env_vars.get("OPENAI_API_KEY")
        
#         # Store in environment (for other components)
#         os.environ["PINECONE_API_KEY"] = api_key
#         os.environ["OPENAI_API_KEY"] = openai_api_key
        
#         # Print debug info about environment variables
#         logger.info(f"PINECONE_API_KEY exists and has length: {len(api_key) if api_key else 0}")
#         logger.info(f"Pinecone API key starts with: {api_key[:10] if api_key and len(api_key) > 10 else 'N/A'}")
#         logger.info(f"OPENAI_API_KEY exists and has length: {len(openai_api_key) if openai_api_key else 0}")
        
#         if not api_key:
#             raise ValueError("PINECONE_API_KEY not found in .env file")
#         if not openai_api_key:
#             raise ValueError("OPENAI_API_KEY not found in .env file")
        
#         # Initialize Pinecone with the API key
#         self.pc = Pinecone(api_key=api_key)
#         self.index_name = "nvidia-reports"
        
#         # Use OpenAI embeddings
#         self.embedding_model = OpenAIEmbeddings(
#             model="text-embedding-3-small",  # Use small model to match 1536 dimensions
#             openai_api_key=openai_api_key
#         )
#         self.llm = ChatOpenAI(temperature=0, api_key=openai_api_key)
        
#         # Log available indexes for debugging
#         logger.info(f"Available Pinecone indexes: {self.pc.list_indexes()}")
        
#     def query(self, query_text: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> Dict[str, Any]:
#         """
#         Query the RAG system with optional metadata filtering by multiple years and quarters.
        
#         Args:
#             query_text: The query text
#             years: Optional list of years to filter by
#             quarters: Optional list of quarters to filter by
            
#         Returns:
#             Dictionary with retrieved context and generated response
#         """
#         try:
#             # Log query parameters
#             logger.info(f"RAG Query: '{query_text}', Years: {years}, Quarters: {quarters}")
            
#             # Generate embedding for the query using OpenAI
#             query_embedding = self.embedding_model.embed_query(query_text)
            
#             # Prepare metadata filter - convert to format stored in Pinecone
#             filter_dict = {}
#             if years is not None and len(years) > 0:
#                 filter_dict["year"] = {"$in": [str(year) for year in years]}  # Filter for any of the years
#             if quarters is not None and len(quarters) > 0:
#                 filter_dict["quarter"] = {"$in": [f"q{quarter}" for quarter in quarters]}  # Filter for any of the quarters
            
#             logger.info(f"Using filter: {filter_dict}")
            
#             # Connect to index
#             index = self.pc.Index(self.index_name)
            
#             # Perform hybrid search with metadata filtering
#             search_results = index.query(
#                 vector=query_embedding,
#                 filter=filter_dict if filter_dict else None,
#                 top_k=10,
#                 include_metadata=True,
#                 alpha=0.5  # Hybrid search parameter - balance between metadata and vector similarity
#             )
            
#             # Log search results for debugging
#             logger.info(f"Found {len(search_results.matches)} matches")
            
#             # Extract retrieved contexts
#             contexts = []
#             for i, match in enumerate(search_results.matches):
#                 # Extract text and metadata
#                 text = match.metadata.get("text", "")
#                 source = match.metadata.get("source", "Unknown source")
#                 doc_year = match.metadata.get("year", "Unknown year")
#                 doc_quarter = match.metadata.get("quarter", "Unknown quarter")
                
#                 # Log match details for debugging
#                 logger.info(f"Match {i+1}: Score {match.score}, Source: {source}, Year: {doc_year}, Quarter: {doc_quarter}")
                
#                 # Add formatted context
#                 contexts.append(f"[Source: {source}, Year: {doc_year}, Quarter: {doc_quarter}]\n{text}")
            
#             # If no contexts were retrieved, provide a fallback
#             if not contexts:
#                 logger.warning("No relevant information found in Pinecone index")
#                 return {
#                     "context": "",
#                     "response": "I couldn't find any relevant information about NVIDIA based on your query and filters. Please try a different query or adjust the year/quarter filters.",
#                     "sources": []
#                 }
            
#             # Combine contexts for the LLM
#             combined_context = "\n\n".join(contexts)
            
#             # Create prompt for generation
#             prompt = ChatPromptTemplate.from_messages([
#                 ("system", """
#                 You are a financial research assistant specializing in NVIDIA.
#                 Use the following historical information from NVIDIA quarterly reports to answer the query.
#                 Only use the information provided in the context.
#                 If the information is not in the context, say so clearly.
#                 Be specific about any financial figures, percentages, or trends mentioned in the context.
#                 Format your response in a clear, concise manner suitable for financial analysis.
#                 """),
#                 ("human", "Context information:\n{context}\n\nQuery: {query}")
#             ])
            
#             # Generate response using LLM
#             logger.info("Generating response with LLM")
#             chain = prompt | self.llm
#             response = chain.invoke({
#                 "context": combined_context,
#                 "query": query_text
#             })
            
#             logger.info("LLM response generated successfully")
            
#             # Return results
#             return {
#                 "context": combined_context,
#                 "response": response.content,
#                 "sources": [match.metadata.get("source", "Unknown") for match in search_results.matches]
#             }
#         except Exception as e:
#             # Add better error handling
#             logger.error(f"Error in RAG agent: {str(e)}", exc_info=True)
#             return {
#                 "context": "",
#                 "response": f"Error retrieving information: {str(e)}",
#                 "sources": []
#             }
    
#     def test_connection(self) -> bool:
#         """Test connection to Pinecone and verify index exists"""
#         try:
#             indexes = self.pc.list_indexes()
#             logger.info(f"Available indexes: {indexes}")
            
#             # Check if our index exists in the available indexes
#             if self.index_name not in [idx.name for idx in indexes]:
#                 logger.error(f"Index '{self.index_name}' not found in available indexes: {indexes}")
#                 return False
                
#             # Test a simple query to make sure we can connect
#             index = self.pc.Index(self.index_name)
#             stats = index.describe_index_stats()
#             logger.info(f"Index stats: {stats}")
            
#             return True
#         except Exception as e:
#             logger.error(f"Error testing Pinecone connection: {str(e)}", exc_info=True)
#             return False



# backend/agents/enhanced_rag_agent.py
import os
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import time
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagAgent:
    """
    Enhanced RAG Agent with improved prompting, guardrails, and structure.
    
    Features:
    - Multi-step retrieval process with query transformation
    - Structured output validation
    - Citation tracking for all claims
    - Semantic chunking for better context retrieval
    - Reliability metrics and self-evaluation
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the Enhanced RAG Agent.
        
        Args:
            verbose: Whether to enable verbose logging
        """
        # Configure logging
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Simple direct environment loading
        load_dotenv()
        
        # Get API keys from environment
        api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Validate keys are present
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        # Log for debugging
        logger.info(f"PINECONE_API_KEY exists and has length: {len(api_key) if api_key else 0}")
        logger.info(f"Pinecone API key starts with: {api_key[:5] if api_key and len(api_key) > 5 else 'N/A'}")
        logger.info(f"OPENAI_API_KEY exists and has length: {len(openai_api_key) if openai_api_key else 0}")
        
        # Initialize Pinecone with the API key
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "nvidia-reports"
        
        # Use OpenAI embeddings
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Use small model to match 1536 dimensions
            openai_api_key=openai_api_key
        )
        
        # Initialize LLMs with different settings for different tasks
        self.llm = ChatOpenAI(
            temperature=0, 
            api_key=openai_api_key,
            model="gpt-4-0125-preview",  # Latest GPT-4 model for best performance
            max_tokens=3500  # Allow for longer, more detailed responses
        )
        
        # Lower temperature LLM for query rewriting
        self.query_llm = ChatOpenAI(
            temperature=0, 
            api_key=openai_api_key,
            model="gpt-4-0125-preview",
        )
        
        # Define output schemas for structured responses
        self.response_schemas = [
            ResponseSchema(
                name="analysis",
                description="The comprehensive analysis based on the NVIDIA report data",
                type="string"
            ),
            ResponseSchema(
                name="citations",
                description="A list of citations for each claim made in the analysis, formatted as " + 
                          "[source, year, quarter] pairs",
                type="array"
            ),
            ResponseSchema(
                name="confidence_score",
                description="A score from 0 to 100 indicating how confident the system is in the analysis " +
                           "based on the retrieved context",
                type="integer"
            ),
            ResponseSchema(
                name="missing_info",
                description="Information that would be helpful to answer the query but is missing from the context",
                type="string"
            )
        ]
        
        # Output parser for structured responses
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        
        # Create format instructions
        self.format_instructions = self.output_parser.get_format_instructions()
        
        # Log available indexes for debugging
        if self.verbose:
            logger.debug(f"Available Pinecone indexes: {self.pc.list_indexes()}")
    
    def rewrite_query(self, original_query: str, years: Optional[List[int]] = None, 
                     quarters: Optional[List[int]] = None) -> str:
        """
        Rewrite and expand the query to improve retrieval performance.
        
        Args:
            original_query: The original query
            years: Optional list of years to include in query expansion
            quarters: Optional list of quarters to include in query expansion
            
        Returns:
            Expanded query
        """
        try:
            # Create date context string
            date_context = ""
            if years:
                date_context += f"Years of interest: {', '.join(map(str, years))}. "
            if quarters:
                date_context += f"Quarters of interest: {', '.join(map(str, quarters))}. "
            
            # Create prompt for query expansion
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a query expansion expert for a financial RAG system focusing on NVIDIA reports.
                Your task is to rewrite the given query to make it more effective for semantic search.
                
                Guidelines for query enhancement:
                
                1. Identify key financial metrics, terms, and concepts related to the query
                2. Include relevant synonyms and related terms investors might look for
                3. Expand acronyms into their full forms with both versions
                4. Add important context around product lines, market segments, or technologies
                5. Include financial terminology that would appear in NVIDIA quarterly reports
                6. Format the expanded query as a natural language question or request
                7. Keep the expanded query focused and relevant to the original intent
                8. Do not add speculative information or make assumptions
                9. Incorporate any specific time periods (years/quarters) mentioned

                Provide only the rewritten query, nothing else.
                """),
                ("human", """
                Original query: {original_query}
                {date_context}
                
                Rewritten query:
                """)
            ])
            
            # Generate expanded query
            chain = LLMChain(llm=self.query_llm, prompt=prompt)
            result = chain.run(original_query=original_query, date_context=date_context)
            
            # Clean up response
            expanded_query = result.strip()
            logger.info(f"Expanded query: {expanded_query}")
            
            return expanded_query
            
        except Exception as e:
            logger.error(f"Error during query rewriting: {str(e)}", exc_info=True)
            # Fall back to original query on error
            return original_query
    
    def retrieve_context(self, query_text: str, expanded_query: str, 
                        years: Optional[List[int]] = None, 
                        quarters: Optional[List[int]] = None,
                        top_k: int = 15) -> Tuple[List[Dict], List[str]]:
        """
        Retrieve context from the vector store using hybrid search and metadata filtering.
        
        Args:
            query_text: The original query text
            expanded_query: The expanded query text
            years: Optional list of years to filter by
            quarters: Optional list of quarters to filter by
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (contexts, sources)
        """
        try:
            # Generate embedding for the expanded query using OpenAI
            query_embedding = self.embedding_model.embed_query(expanded_query)
            
            # Prepare metadata filter - convert to format stored in Pinecone
            filter_dict = {}
            if years is not None and len(years) > 0:
                filter_dict["year"] = {"$in": [str(year) for year in years]}  # Filter for any of the years
            if quarters is not None and len(quarters) > 0:
                filter_dict["quarter"] = {"$in": [f"q{quarter}" for quarter in quarters]}  # Filter for any of the quarters
            
            if self.verbose:
                logger.debug(f"Using filter: {filter_dict}")
            
            # Connect to index
            index = self.pc.Index(self.index_name)
            
            # Perform hybrid search with metadata filtering
            search_results = index.query(
                vector=query_embedding,
                filter=filter_dict if filter_dict else None,
                top_k=top_k,
                include_metadata=True,
                alpha=0.3  # Hybrid search parameter - adjust to balance between vector and metadata 
            )
            
            if self.verbose:
                logger.debug(f"Found {len(search_results.matches)} matches")
            
            # Extract retrieved contexts with metadata
            contexts = []
            sources = []
            
            for i, match in enumerate(search_results.matches):
                # Extract text and metadata
                text = match.metadata.get("text", "")
                source = match.metadata.get("source", "Unknown source")
                doc_year = match.metadata.get("year", "Unknown year")
                doc_quarter = match.metadata.get("quarter", "Unknown quarter")
                
                # Add formatted context with metadata
                context_entry = {
                    "text": text,
                    "source": source,
                    "year": doc_year,
                    "quarter": doc_quarter,
                    "relevance_score": match.score
                }
                contexts.append(context_entry)
                
                # Add to sources list
                source_entry = f"{source} ({doc_year}, {doc_quarter})"
                if source_entry not in sources:
                    sources.append(source_entry)
                
                # Log match details for debugging
                if self.verbose:
                    logger.debug(f"Match {i+1}: Score {match.score}, Source: {source}, Year: {doc_year}, Quarter: {doc_quarter}")
            
            return contexts, sources
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
            return [], []

    def format_context_for_llm(self, contexts: List[Dict]) -> str:
        """
        Format retrieved contexts for input to the LLM.
        
        Args:
            contexts: List of context entries with metadata
            
        Returns:
            Formatted context string
        """
        if not contexts:
            return ""
        
        # Sort contexts by relevance score
        sorted_contexts = sorted(contexts, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Format each context with metadata
        formatted_contexts = []
        for i, context in enumerate(sorted_contexts):
            text = context.get("text", "")
            source = context.get("source", "Unknown source")
            year = context.get("year", "Unknown year")
            quarter = context.get("quarter", "Unknown quarter")
            
            formatted_context = f"[Context {i+1}]\n"
            formatted_context += f"Source: {source}\n"
            formatted_context += f"Year: {year}\n"
            formatted_context += f"Quarter: {quarter}\n"
            formatted_context += f"Content: {text}\n"
            formatted_contexts.append(formatted_context)
        
        # Combine all contexts
        return "\n\n".join(formatted_contexts)

    def generate_response(self, query_text: str, context: str) -> Dict:
        """
        Generate a structured response using the LLM.
        
        Args:
            query_text: The query text
            context: The formatted context
            
        Returns:
            Structured response with analysis, citations, confidence score and missing info
        """
        try:
            # Create enhanced prompt for generation
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a financial research analyst specializing in NVIDIA, tasked with providing accurate, detailed information based on the company's quarterly reports and financial data.

                ## TASK:
                Analyze the provided context information from NVIDIA reports to answer the query comprehensively, creating detailed financial analysis suitable for professional investors.

                ## CONTENT GUIDELINES:
                1. ACCURACY: Only use information explicitly stated in the provided context. Never fabricate data, figures, or quotes.
                2. CITATIONS: Cite the source document, year, and quarter for every factual claim you make.
                3. COMPLETENESS: Address all aspects of the query that can be answered from the context.
                4. FINANCIAL FOCUS: Emphasize specific financial metrics with exact figures including:
                   - Revenue figures (total and segment-specific)
                   - Growth rates (YoY and QoQ percentages)
                   - Profit margins
                   - EPS figures
                   - Market share statistics
                   - R&D expenditures
                5. ANALYTICAL DEPTH: Provide interpretation of the numbers, not just raw data.
                6. TEMPORAL CLARITY: Clearly distinguish between historical results, current status, and forward-looking statements.
                7. SPECIFICITY: Include precise dates, dollar amounts, percentages, and growth rates whenever available.
                8. COMPREHENSIVENESS: Cover multiple aspects of NVIDIA's business relevant to the query.

                ## FORMATTING REQUIREMENTS:
                1. Create a well-structured, detailed analysis (500-750 words)
                2. Use markdown formatting consistently (headings, bold, bullet points)
                3. Use tables when presenting comparative data if appropriate
                4. Bold all key metrics and important figures (revenues, profits, percentages)
                5. Organize information logically with clear section headings
                6. Create a proper citations section at the end listing all sources

                ## QUALITY STANDARDS:
                1. PROFESSIONAL TONE: Maintain formal, analytical language appropriate for financial research
                2. BALANCED PERSPECTIVE: Present both strengths and challenges revealed in the data
                3. CONTEXT-AWARENESS: Connect individual metrics to broader business and industry trends
                4. LIMITATIONS: Clearly indicate if information needed to fully answer the query is missing
                5. CONFIDENCE: Provide a confidence score (0-100) based on how well the context supports your answer

                ## OUTPUT FORMAT:
                {format_instructions}
                """),
                ("human", """
                Query: {query}
                
                Context information:
                {context}
                """)
            ])
            
            # Generate response using LLM
            logger.info("Generating structured response with LLM")
            chain = prompt | self.llm
            result = chain.invoke({
                "query": query_text,
                "context": context,
                "format_instructions": self.format_instructions
            })
            
            # Parse the structured output
            try:
                parsed_output = self.output_parser.parse(result.content)
                logger.info("Successfully parsed structured output")
                return parsed_output
            except Exception as e:
                logger.error(f"Error parsing structured output: {str(e)}", exc_info=True)
                # Return a fallback response structure if parsing fails
                return {
                    "analysis": result.content,
                    "citations": [],
                    "confidence_score": 0,
                    "missing_info": "Error parsing structured output"
                }
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {
                "analysis": f"Error generating response: {str(e)}",
                "citations": [],
                "confidence_score": 0,
                "missing_info": "An error occurred during processing"
            }
    
    def verify_output(self, output: Dict, contexts: List[Dict]) -> Dict:
        """
        Verify that the generated output is supported by the context.
        
        Args:
            output: The generated output
            contexts: The retrieved contexts
            
        Returns:
            Verified and possibly adjusted output
        """
        try:
            # Check if confidence score is reasonable
            if output.get("confidence_score", 0) > 90 and output.get("missing_info", ""):
                # Adjust confidence if missing information is noted
                output["confidence_score"] = min(output["confidence_score"], 85)
                
            # Ensure we have citations
            if not output.get("citations", []) and output.get("analysis", ""):
                logger.warning("No citations provided in the output")
                
                # Generate generic citations based on contexts
                generic_citations = []
                for context in contexts[:5]:  # Use top 5 contexts
                    source = context.get("source", "Unknown")
                    year = context.get("year", "Unknown")
                    quarter = context.get("quarter", "Unknown")
                    generic_citations.append([source, year, quarter])
                
                output["citations"] = generic_citations
                
            return output
                
        except Exception as e:
            logger.error(f"Error verifying output: {str(e)}", exc_info=True)
            return output
    
    def format_final_response(self, output: Dict, sources: List[str]) -> Dict:
        """
        Format the final response for delivery to the user.
        
        Args:
            output: The verified output
            sources: The sources used
            
        Returns:
            Formatted response for the API
        """
        try:
            # Get the analysis text
            analysis = output.get("analysis", "")
            
            # Check if we need to format it further
            if not "##" in analysis and not "**" in analysis:
                # Add some basic formatting to improve readability
                analysis = "## NVIDIA Financial Analysis\n\n" + analysis
                
                # Try to identify paragraphs for section headings
                paragraphs = analysis.split("\n\n")
                formatted_paragraphs = []
                
                current_section = "Overview"
                for i, para in enumerate(paragraphs):
                    if i == 0:
                        # First paragraph is already under the main heading
                        formatted_paragraphs.append(para)
                    elif len(para) > 0 and para[0].isupper() and len(para.split()) <= 5:
                        # This looks like a potential section title
                        current_section = para
                        formatted_paragraphs.append(f"### {current_section}")
                    else:
                        # Regular paragraph
                        formatted_paragraphs.append(para)
                        
                # Reconstruct with new formatting
                analysis = "\n\n".join(formatted_paragraphs)
                
                # Add bold to numbers and percentages
                import re
                analysis = re.sub(r'(\$[\d,]+\.?\d*|\d+\.\d+%|\d+%)', r'**\1**', analysis)
            
            # Get other components
            citations = output.get("citations", [])
            confidence_score = output.get("confidence_score", 0)
            missing_info = output.get("missing_info", "")
            
            # Format citations section at the end
            if citations:
                citation_text = "\n\n## Sources and References\n\n"
                added_citations = set()
                
                for citation in citations:
                    if len(citation) >= 3:
                        source, year, quarter = citation[0], citation[1], citation[2]
                        citation_key = f"{source}-{year}-{quarter}"
                        
                        if citation_key not in added_citations:
                            citation_text += f"- {source} ({year}, {quarter})\n"
                            added_citations.add(citation_key)
                
                analysis += citation_text
            
            # Add confidence information
            if confidence_score > 0:
                reliability_level = "High" if confidence_score >= 80 else ("Medium" if confidence_score >= 50 else "Low")
                analysis += f"\n\n## Reliability Assessment\n\nConfidence Level: **{reliability_level}** ({confidence_score}/100)"
            
            # Add missing information if available
            if missing_info and missing_info.lower() not in ["none", "n/a", "nothing"]:
                analysis += f"\n\n## Additional Information Needed\n\n{missing_info}"
            
            # Return the final formatted response
            return {
                "response": analysis,
                "sources": sources,
                "confidence_score": confidence_score
            }
                
        except Exception as e:
            logger.error(f"Error formatting final response: {str(e)}", exc_info=True)
            return {
                "response": output.get("analysis", f"Error formatting response: {str(e)}"),
                "sources": sources,
                "confidence_score": output.get("confidence_score", 0)
            }
    
    def query(self, query_text: str, years: Optional[List[int]] = None, 
              quarters: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline with guardrails and structured output.
        
        Args:
            query_text: The query text
            years: Optional list of years to filter by
            quarters: Optional list of quarters to filter by
            
        Returns:
            Dictionary with retrieved context, generated response, and sources
        """
        try:
            start_time = time.time()
            logger.info(f"RAG Query: '{query_text}', Years: {years}, Quarters: {quarters}")
            
            # Step 1: Rewrite the query to improve retrieval
            expanded_query = self.rewrite_query(query_text, years, quarters)
            
            # Step 2: Retrieve relevant contexts
            contexts, sources = self.retrieve_context(query_text, expanded_query, years, quarters)
            
            # If no contexts were retrieved, provide a fallback
            if not contexts:
                logger.warning("No relevant information found in Pinecone index")
                return {
                    "context": "",
                    "response": "I couldn't find any relevant information about NVIDIA based on your query and filters. Please try a different query or adjust the year/quarter filters.",
                    "sources": [],
                    "confidence_score": 0
                }
            
            # Step 3: Format contexts for the LLM
            formatted_context = self.format_context_for_llm(contexts)
            
            # Step 4: Generate structured response
            raw_output = self.generate_response(query_text, formatted_context)
            
            # Step 5: Verify and adjust output if needed
            verified_output = self.verify_output(raw_output, contexts)
            
            # Step 6: Format final response
            final_response = self.format_final_response(verified_output, sources)
            
            # Add original context and processing time
            final_response["context"] = formatted_context
            final_response["processing_time"] = f"{time.time() - start_time:.2f}s"
            
            return final_response
            
        except Exception as e:
            # Add better error handling
            logger.error(f"Error in Enhanced RAG agent: {str(e)}", exc_info=True)
            return {
                "context": "",
                "response": f"Error retrieving information: {str(e)}",
                "sources": [],
                "confidence_score": 0
            }
    
    def test_connection(self) -> bool:
        """Test connection to Pinecone and verify index exists"""
        try:
            indexes = self.pc.list_indexes()
            logger.info(f"Available indexes: {indexes}")
            
            # Check if our index exists in the available indexes
            if self.index_name not in [idx.name for idx in indexes]:
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
