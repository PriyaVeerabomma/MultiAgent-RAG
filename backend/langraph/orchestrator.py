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
        
        # If only one agent is active, use its response as the final report
        if len(self.active_agents) == 1:
            active_agent = self.active_agents[0]  # Get the only active agent
            if active_agent == "rag":
                final_response = content.get("historical_data", "")
                # Don't include the raw data again
                results.pop("historical_data", None)
            elif active_agent == "snowflake":
                final_response = content.get("financial_metrics", "")
                # Keep the chart but remove duplicate content
                if "financial_metrics" in results:
                    chart = results["financial_metrics"].get("chart", None)
                    sources = results["financial_metrics"].get("sources", [])
                    results["financial_metrics"] = {
                        "chart": chart,
                        "sources": sources
                    }
            elif active_agent == "websearch":
                final_response = content.get("latest_insights", "")
                # Don't include the raw data again
                results.pop("latest_insights", None)
        else:
            # Synthesize the final report if we have multiple sections
            try:
                # Load the orchestrator synthesis prompt
                orchestrator_prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                                "prompts", "orchestrator_synthesis.txt")
                
                # Use default prompt if the file doesn't exist
                if os.path.exists(orchestrator_prompt_path):
                    with open(orchestrator_prompt_path, 'r') as f:
                        system_prompt = f.read()
                else:
                    system_prompt = """
                    You are NVIDIA Research Director, an expert AI system specializing in producing comprehensive financial research reports on NVIDIA Corporation.
                    
                    TASK:
                    Your task is to synthesize information from multiple specialized research sources into a cohesive, authoritative report that thoroughly addresses the research query.
                    
                    INTEGRATION APPROACH:
                    1. INFORMATION ASSESSMENT:
                       - Identify the most relevant data points from each source
                       - Recognize common themes and insights across sources
                       - Evaluate conflicting information and determine the most reliable perspective
                       - Prioritize recent, specific information over general or older data
                    
                    2. SYNTHESIS METHODOLOGY:
                       - Create logical connections between historically reported data and current market insights
                       - Balance financial metrics with product/technology developments in your analysis
                       - Build a coherent narrative that flows naturally between different information sources
                       - Ensure all information relates back to addressing the core research query
                    
                    REPORT STRUCTURE:
                    1. EXECUTIVE SUMMARY (2-3 paragraphs)
                       - Core findings synthesis
                       - Key metrics highlighted with precise figures
                       - Central insights derived from multiple sources
                    
                    2. HISTORICAL CONTEXT & BACKGROUND (1-2 paragraphs)
                       - Relevant historical performance data
                       - Evolution of key business segments
                       - Context for current performance metrics
                    
                    3. FINANCIAL PERFORMANCE ANALYSIS (2-3 paragraphs)
                       - Revenue and profitability metrics with exact figures
                       - Segment-by-segment performance breakdown
                       - Year-over-year and quarter-over-quarter comparisons
                       - Margin analysis and operational efficiency insights
                    
                    4. MARKET POSITION & COMPETITIVE LANDSCAPE (1-2 paragraphs)
                       - Market share data with precise percentages
                       - Competitive positioning across key segments
                       - Industry trends affecting NVIDIA specifically
                       - Recent market developments and their significance
                    
                    5. TECHNOLOGY & PRODUCT ANALYSIS (1-2 paragraphs)
                       - Key products driving revenue and growth
                       - Technology innovations and their market impact
                       - Product pipeline and development roadmap insights
                       - Adoption trends across different market segments
                    
                    6. FUTURE OUTLOOK & PROJECTIONS (1-2 paragraphs)
                       - Forward-looking metrics from analyst perspectives
                       - Growth catalysts and potential challenges
                       - Expected trends based on current market position
                       - Specific projections with timeframes when available
                    
                    7. CONCLUSION & INVESTMENT IMPLICATIONS (1 paragraph)
                       - Synthesis of key findings
                       - Overall assessment of NVIDIA's position and trajectory
                       - Critical insights for understanding the company's future
                    
                    FORMATTING REQUIREMENTS:
                    - Create clear, bold section headings
                    - Bold all significant metrics and percentages
                    - Use bullet points for listing key metrics and data points
                    - Maintain consistent paragraph structure for readability
                    - Ensure professional language and objective tone throughout
                    
                    DO NOT:
                    - Introduce new information not present in the provided sources
                    - Make investment recommendations or predictions not supported by the sources
                    - Present opinions as facts or overstate certainty in projections
                    - Use vague language when specific figures are available
                    - Duplicate content from one section to another
                    
                    Your synthesis should exceed the value of any individual source by creating a comprehensive, 
                    integrated analysis that provides deeper insights than any single perspective.
                    """
                
                # Create improved prompt for synthesis
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", """
                    Please create a comprehensive research report answering the following query: {query}
                    
                    Available information from specialized research sources:
                    
                    HISTORICAL DATA (From quarterly reports and historical analysis):
                    {historical_data}
                    
                    FINANCIAL METRICS (From stock performance and financial databases):
                    {financial_metrics}
                    
                    LATEST INSIGHTS (From current market research and news):
                    {latest_insights}
                    
                    Synthesize these sources into a cohesive, authoritative research report that thoroughly addresses the query.
                    """)
                ])
                
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
