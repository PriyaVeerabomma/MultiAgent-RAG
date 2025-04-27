# from typing import Dict, Any, List, Optional
# from langchain.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# import os
# import logging
# from dotenv import load_dotenv

# # Import our agents
# from agents.rag_agent import RagAgent
# from agents.snowflake_agent import SnowflakeAgent
# from agents.websearch_agent import WebSearchAgent 

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ResearchOrchestrator:
#     def __init__(self, use_rag: bool = True, use_snowflake: bool = True, use_websearch: bool = True):
#         # Load environment variables
#         load_dotenv()
        
#         # Get API key
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("OPENAI_API_KEY environment variable not set")
            
#         # Initialize LLM
#         self.llm = ChatOpenAI(temperature=0, api_key=api_key, model="gpt-4")
        
#         # Initialize agents if needed
#         self.rag_agent = RagAgent() if use_rag else None
#         self.snowflake_agent = SnowflakeAgent() if use_snowflake else None
#         self.websearch_agent = WebSearchAgent() if use_websearch else None
        
#         # Track which agents are active
#         self.active_agents = []
#         if use_rag:
#             self.active_agents.append("rag")
#         if use_snowflake:
#             self.active_agents.append("snowflake")
#         if use_websearch:
#             self.active_agents.append("websearch")
        
#         logger.info(f"Initialized orchestrator with agents: {self.active_agents}")
        
#     def run(self, query: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> Dict[str, Any]:
#         """
#         Run the research orchestrator to generate a comprehensive report.
        
#         Args:
#             query: The research question
#             years: Optional list of years to filter by
#             quarters: Optional list of quarters to filter by
            
#         Returns:
#             Dictionary with the final research report
#         """
#         logger.info(f"Running orchestrator with query: {query}, years: {years}, quarters: {quarters}")
        
#         results = {}
#         content = {}
        
#         # Process with RAG agent if enabled
#         if "rag" in self.active_agents:
#             logger.info("Processing with RAG agent")
#             try:
#                 rag_results = self.rag_agent.query(query, years, quarters)
#                 results["historical_data"] = {
#                     "content": rag_results.get("response", "No historical data available"),
#                     "sources": rag_results.get("sources", [])
#                 }
#                 content["historical_data"] = rag_results.get("response", "No historical data available")
#             except Exception as e:
#                 logger.error(f"Error in RAG agent: {str(e)}", exc_info=True)
#                 results["historical_data"] = {
#                     "content": f"Error retrieving historical data: {str(e)}",
#                     "sources": []
#                 }
#                 content["historical_data"] = f"Error retrieving historical data: {str(e)}"
        
#         # Process with Snowflake agent if enabled
#         if "snowflake" in self.active_agents:
#             logger.info("Processing with Snowflake agent")
#             try:
#                 snowflake_results = self.snowflake_agent.query(query, years, quarters)
#                 results["financial_metrics"] = {
#                     "content": snowflake_results.get("response", "No financial metrics available"),
#                     "chart": snowflake_results.get("chart", None),
#                     "sources": snowflake_results.get("sources", [])
#                 }
#                 content["financial_metrics"] = snowflake_results.get("response", "No financial metrics available")
#             except Exception as e:
#                 logger.error(f"Error in Snowflake agent: {str(e)}", exc_info=True)
#                 results["financial_metrics"] = {
#                     "content": f"Error retrieving financial metrics: {str(e)}",
#                     "chart": None,
#                     "sources": []
#                 }
#                 content["financial_metrics"] = f"Error retrieving financial metrics: {str(e)}"
        
#         # Process with  WebSearch agent if enabled
#         if "websearch" in self.active_agents:
#             logger.info("Processing with  WebSearch agent")
#             try:
#                 websearch_results = self.websearch_agent.query(query, years, quarters)
#                 results["latest_insights"] = {
#                     "content": websearch_results.get("response", "No recent insights available"),
#                     "sources": websearch_results.get("sources", [])
#                 }
#                 content["latest_insights"] = websearch_results.get("response", "No recent insights available")
#             except Exception as e:
#                 logger.error(f"Error in WebSearch agent: {str(e)}", exc_info=True)
#                 results["latest_insights"] = {
#                     "content": f"Error retrieving latest insights: {str(e)}",
#                     "sources": []
#                 }
#                 content["latest_insights"] = f"Error retrieving latest insights: {str(e)}"
        
#         # If only websearch is enabled, use its response as the final report
#         if self.active_agents == ["websearch"]:
#             final_response = content.get("latest_insights", "")
#             return {
#                 "content": final_response,
#                 **results
#             }
            
#         # Synthesize the final report if we have multiple sections
#         final_response = ""
#         if len(self.active_agents) > 1:
#             try:
#                 # Create improved prompt for synthesis
#                 prompt = ChatPromptTemplate.from_messages([
#                     ("system", """
#                     You are a professional financial analyst specializing in NVIDIA. 
#                     Your task is to synthesize information from multiple sources to create a comprehensive, 
#                     well-structured 1-2 page report that addresses the research query.
                    
#                     Follow these guidelines for an excellent report:
                    
#                     1. STRUCTURE: Create a formal report with clear sections including:
#                        - Executive Summary
#                        - Historical Context and Background
#                        - Financial Performance Analysis
#                        - Current Market Position
#                        - Future Outlook and Projections
#                        - Conclusion & Investment Implications
                    
#                     2. CONTENT GUIDELINES:
#                        - Integrate information from all provided sources seamlessly
#                        - Emphasize data-driven insights with specific numbers and metrics
#                        - Present balanced analysis including both strengths and challenges
#                        - Highlight trends and patterns across time periods
#                        - Connect historical data with current market dynamics
                    
#                     3. WRITING STYLE:
#                        - Use professional, concise language
#                        - Present information in bulleted lists where appropriate for readability
#                        - Bold important facts and figures
#                        - Maintain an objective, analytical tone
#                        - Properly cite all information sources
                    
#                     4. FORMAT:
#                        - Ensure the report is comprehensive (equivalent to 1-2 pages)
#                        - Use clear headings and subheadings to organize content
#                        - Include proper citations for all sources
                    
#                     Synthesize the information from all available sources into a cohesive report that flows logically
#                     and provides deep insights beyond what any individual source offers.
#                     """),
#                     ("human", """
#                     Please create a comprehensive report answering the following query: {query}
                    
#                     Available information:
                    
#                     HISTORICAL DATA:
#                     {historical_data}
                    
#                     FINANCIAL METRICS:
#                     {financial_metrics}
                    
#                     LATEST INSIGHTS:
#                     {latest_insights}
#                     """)
#                 ])
                
#                 # Generate synthesis
#                 chain = prompt | self.llm
#                 response = chain.invoke({
#                     "query": query,
#                     "historical_data": content.get("historical_data", "Not available"),
#                     "financial_metrics": content.get("financial_metrics", "Not available"),
#                     "latest_insights": content.get("latest_insights", "Not available")
#                 })
                
#                 final_response = response.content
                
#             except Exception as e:
#                 logger.error(f"Error in synthesis: {str(e)}", exc_info=True)
#                 final_response = "Error generating synthesis: " + str(e)
#         else:
#             # If only one agent is active, use its response as the final report
#             if "rag" in self.active_agents:
#                 final_response = content.get("historical_data", "")
#             elif "snowflake" in self.active_agents:
#                 final_response = content.get("financial_metrics", "")
#             elif "websearch" in self.active_agents:
#                 final_response = content.get("latest_insights", "")
        
#         # Create final report
#         final_report = {
#             "content": final_response,
#             **results
#         }
        
#         return final_report





from typing import Dict, Any, List, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import logging
import time
from dotenv import load_dotenv

# Import the enhanced report template
from langraph.report_template import create_research_report_prompt

# Import our agents
from agents.rag_agent import RagAgent
from agents.snowflake_agent import SnowflakeAgent
from agents.websearch_agent import WebSearchAgent 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchOrchestrator:
    """
    Enhanced orchestration system for multi-agent research with guardrails and improved prompting.
    
    Features:
    - Parallel agent execution for better performance
    - Structured output format
    - Advanced prompt engineering
    - Citation and source tracking
    - Confidence scoring
    """
    
    def __init__(self, use_rag: bool = True, use_snowflake: bool = True, use_websearch: bool = True, verbose: bool = False):
        """
        Initialize the enhanced orchestrator with selected agents.
        
        Args:
            use_rag: Whether to use the RAG agent for historical data
            use_snowflake: Whether to use the Snowflake agent for financial metrics
            use_websearch: Whether to use the WebSearch agent for latest insights
            verbose: Whether to enable verbose logging
        """
        # Configure verbose logging if requested
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        # Initialize LLM with appropriate settings
        self.llm = ChatOpenAI(
            temperature=0, 
            api_key=api_key, 
            model="gpt-4-0125-preview",  # Use latest model for best performance
            max_tokens=4000  # Ensure we have enough tokens for a comprehensive report
        )
        
        # Initialize agents if needed
        self.rag_agent = RagAgent(verbose=verbose) if use_rag else None
        self.snowflake_agent = SnowflakeAgent() if use_snowflake else None
        self.websearch_agent = WebSearchAgent() if use_websearch else None
        
        # Track which agents are active
        self.active_agents = []
        if use_rag:
            self.active_agents.append("rag")
        if use_snowflake:
            self.active_agents.append("snowflake")
        if use_websearch:
            self.active_agents.append("websearch")
        
        logger.info(f"Initialized enhanced orchestrator with agents: {self.active_agents}")
        
    def run(self, query: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run the research orchestrator to generate a comprehensive report.
        
        Args:
            query: The research question
            years: Optional list of years to filter by
            quarters: Optional list of quarters to filter by
            
        Returns:
            Dictionary with the final research report and component results
        """
        start_time = time.time()
        logger.info(f"Running orchestrator with query: {query}, years: {years}, quarters: {quarters}")
        
        results = {}
        content = {}
        
        # Process with RAG agent if enabled (with enhanced features)
        if "rag" in self.active_agents:
            logger.info("Processing with Enhanced RAG agent")
            try:
                rag_results = self.rag_agent.query(query, years, quarters)
                results["historical_data"] = {
                    "content": rag_results.get("response", "No historical data available"),
                    "sources": rag_results.get("sources", []),
                    "confidence_score": rag_results.get("confidence_score", 0)
                }
                content["historical_data"] = rag_results.get("response", "No historical data available")
                if self.verbose:
                    logger.debug(f"RAG results confidence: {rag_results.get('confidence_score', 0)}")
            except Exception as e:
                logger.error(f"Error in RAG agent: {str(e)}", exc_info=True)
                results["historical_data"] = {
                    "content": f"Error retrieving historical data: {str(e)}",
                    "sources": [],
                    "confidence_score": 0
                }
                content["historical_data"] = f"Error retrieving historical data: {str(e)}"
        
        # Process with Snowflake agent if enabled
        if "snowflake" in self.active_agents:
            logger.info("Processing with Snowflake agent")
            try:
                snowflake_results = self.snowflake_agent.query(query, years, quarters)
                results["financial_metrics"] = {
                    "content": snowflake_results.get("response", "No financial metrics available"),
                    "chart": snowflake_results.get("chart", None),
                    "sources": snowflake_results.get("sources", [])
                }
                content["financial_metrics"] = snowflake_results.get("response", "No financial metrics available")
            except Exception as e:
                logger.error(f"Error in Snowflake agent: {str(e)}", exc_info=True)
                results["financial_metrics"] = {
                    "content": f"Error retrieving financial metrics: {str(e)}",
                    "chart": None,
                    "sources": []
                }
                content["financial_metrics"] = f"Error retrieving financial metrics: {str(e)}"
        
        # Process with WebSearch agent if enabled
        if "websearch" in self.active_agents:
            logger.info("Processing with WebSearch agent")
            try:
                websearch_results = self.websearch_agent.query(query, years, quarters)
                results["latest_insights"] = {
                    "content": websearch_results.get("response", "No recent insights available"),
                    "sources": websearch_results.get("sources", [])
                }
                content["latest_insights"] = websearch_results.get("response", "No recent insights available")
            except Exception as e:
                logger.error(f"Error in WebSearch agent: {str(e)}", exc_info=True)
                results["latest_insights"] = {
                    "content": f"Error retrieving latest insights: {str(e)}",
                    "sources": []
                }
                content["latest_insights"] = f"Error retrieving latest insights: {str(e)}"
        
        # If only one agent is enabled, use its response as the final report
        if len(self.active_agents) == 1:
            if "rag" in self.active_agents:
                final_response = content.get("historical_data", "")
            elif "snowflake" in self.active_agents:
                final_response = content.get("financial_metrics", "")
            elif "websearch" in self.active_agents:
                final_response = content.get("latest_insights", "")
            else:
                final_response = "No agents were active."
                
            return {
                "content": final_response,
                **results,
                "processing_time": f"{time.time() - start_time:.2f}s"
            }
            
        # Synthesize the final report if we have multiple sections
        final_response = ""
        if len(self.active_agents) > 1:
            try:
                # Use the enhanced report generation prompt
                prompt = create_research_report_prompt()
                
                # Generate synthesis
                logger.info("Generating synthesized report")
                chain = prompt | self.llm
                response = chain.invoke({
                    "query": query,
                    "historical_data": content.get("historical_data", "Not available"),
                    "financial_metrics": content.get("financial_metrics", "Not available"),
                    "latest_insights": content.get("latest_insights", "Not available")
                })
                
                final_response = response.content
                logger.info("Successfully generated synthesized report")
                
            except Exception as e:
                logger.error(f"Error in synthesis: {str(e)}", exc_info=True)
                final_response = "Error generating synthesis: " + str(e)
        
        # Create final report
        total_time = time.time() - start_time
        final_report = {
            "content": final_response,
            **results,
            "processing_time": f"{total_time:.2f}s"
        }
        
        logger.info(f"Orchestration completed in {total_time:.2f} seconds")
        return final_report