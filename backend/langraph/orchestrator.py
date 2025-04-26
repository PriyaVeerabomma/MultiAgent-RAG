# backend/langraph/orchestrator.py
from typing import Dict, Any, List, Optional, TypedDict
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import logging

# Import our agents
from agents.rag_agent import RagAgent
from agents.snowflake_agent import SnowflakeAgent
from agents.websearch_agent import WebSearchAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchOrchestrator:
    def __init__(self, use_rag: bool = True, use_snowflake: bool = True, use_websearch: bool = True):
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0, api_key=api_key)
        
        # Initialize agents if needed
        self.rag_agent = RagAgent() if use_rag else None
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
        
        logger.info(f"Initialized orchestrator with agents: {self.active_agents}")
        
    def run(self, query: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run the research orchestrator to generate a comprehensive report.
        
        Args:
            query: The research question
            years: Optional list of years to filter by
            quarters: Optional list of quarters to filter by
            
        Returns:
            Dictionary with the final research report
        """
        logger.info(f"Running orchestrator with query: {query}, years: {years}, quarters: {quarters}")
        
        results = {}
        content = {}
        
        # Process with RAG agent if enabled
        if "rag" in self.active_agents:
            logger.info("Processing with RAG agent")
            try:
                rag_results = self.rag_agent.query(query, years, quarters)
                results["historical_data"] = {
                    "content": rag_results.get("response", "No historical data available"),
                    "sources": rag_results.get("sources", [])
                }
                content["historical_data"] = rag_results.get("response", "No historical data available")
            except Exception as e:
                logger.error(f"Error in RAG agent: {str(e)}", exc_info=True)
                results["historical_data"] = {
                    "content": f"Error retrieving historical data: {str(e)}",
                    "sources": []
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
        
        # Synthesize the final report if we have multiple sections
        final_response = ""
        if len(self.active_agents) > 1:
            try:
                # Create prompt for synthesis
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """
                    You are a financial research assistant specialized in NVIDIA. 
                    Synthesize information from multiple sources to create a comprehensive report.
                    Include relevant information from all available sources.
                    Structure your response clearly with logical flow between different types of information.
                    """),
                    ("human", """
                    Please create a comprehensive report answering the following query: {query}
                    
                    Available information:
                    
                    Historical Data: {historical_data}
                    
                    Financial Metrics: {financial_metrics}
                    
                    Latest Insights: {latest_insights}
                    """)
                ])
                
                # Generate synthesis
                chain = prompt | self.llm
                response = chain.invoke({
                    "query": query,
                    "historical_data": content.get("historical_data", "Not available"),
                    "financial_metrics": content.get("financial_metrics", "Not available"),
                    "latest_insights": content.get("latest_insights", "Not available")
                })
                
                final_response = response.content
                
            except Exception as e:
                logger.error(f"Error in synthesis: {str(e)}", exc_info=True)
                final_response = "Error generating synthesis: " + str(e)
        else:
            # If only one agent is active, use its response as the final report
            if "rag" in self.active_agents:
                final_response = content.get("historical_data", "")
            elif "snowflake" in self.active_agents:
                final_response = content.get("financial_metrics", "")
            elif "websearch" in self.active_agents:
                final_response = content.get("latest_insights", "")
        
        # Create final report
        final_report = {
            "content": final_response,
            **results
        }
        
        return final_report