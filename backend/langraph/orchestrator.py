# backend/langraph/orchestrator.py
from typing import Dict, Any, List, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import logging
import re
from dotenv import load_dotenv

# Import our agents
from agents.rag_agent import RagAgent
from agents.snowflake_agent import SnowflakeAgent
from agents.websearch_agent import WebSearchAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchOrchestrator:
    def __init__(self, use_rag: bool = True, use_snowflake: bool = True, use_websearch: bool = True):
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0, api_key=api_key, model="gpt-4")
        
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
    
    def _analyze_query_for_agents(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to determine which agents would be most appropriate
        and how to handle the query.
        
        Args:
            query: The research question
            
        Returns:
            Dictionary with analysis results and agent recommendations
        """
        query_lower = query.lower()
        
        # Check if query is about NVIDIA
        nvidia_terms = [
            r'nvidia', r'nvda', r'jensen huang', r'gpu', r'data center', 
            r'gaming', r'graphics', r'semiconductor', r'chip', r'ai accelerator'
        ]
        mentions_nvidia = any(term in query_lower for term in nvidia_terms)
        
        # Check if query is financial in nature
        financial_terms = [
            r'revenue', r'earnings', r'profit', r'margin', r'income', 
            r'financial', r'stock', r'price', r'growth', r'sales', 
            r'quarter', r'performance', r'market cap', r'valuation',
            r'eps', r'dividend', r'balance sheet', r'cash flow', r'investment',
            r'share', r'investor', r'trading', r'fiscal', r'metric'
        ]
        is_financial = any(term in query_lower for term in financial_terms)
        
        # Check if query is historical in nature
        historical_terms = [
            r'history', r'founded', r'began', r'origin', r'start', 
            r'first', r'initial', r'early', r'background', r'past',
            r'when', r'heritage', r'legacy', r'creation', r'establish',
            r'inception', r'beginning', r'launch', r'formation', r'created'
        ]
        is_historical = any(term in query_lower for term in historical_terms)
        
        # Check if query is about current/recent events
        recent_terms = [
            r'latest', r'current', r'recent', r'new', r'today', 
            r'now', r'update', r'future', r'forecast', r'outlook',
            r'trend', r'development', r'announce', r'launch', r'reveal',
            r'release', r'this year', r'this quarter', r'this month',
            r'next', r'upcoming', r'projection', r'expected', r'anticipated'
        ]
        is_recent = any(term in query_lower for term in recent_terms)
        
        # Check for specific off-topic indicators
        off_topic_terms = [
            r'elon musk', r'tesla', r'spacex', r'meta', r'facebook', 
            r'apple', r'google', r'microsoft', r'amazon', r'intel', r'amd'
        ]
        mentions_off_topic = any(term in query_lower for term in off_topic_terms)
        
        # Determine which agents to use based on query analysis
        recommended_agents = []
        
        # Add Snowflake for financial queries
        if is_financial:
            recommended_agents.append("snowflake")
        
        # Add RAG for historical queries
        if is_historical:
            recommended_agents.append("rag")
        
        # Add WebSearch for recent queries or if nothing else is recommended
        if is_recent or not (is_financial or is_historical):
            recommended_agents.append("websearch")
        
        # If query mentions off-topic subjects but not NVIDIA, websearch can handle it
        if mentions_off_topic and not mentions_nvidia:
            recommended_agents = ["websearch"]
        
        # If no specific patterns matched, use all available agents
        if not recommended_agents:
            recommended_agents = self.active_agents.copy()
        
        # Make sure we only return agents that are actually active
        final_agents = [agent for agent in recommended_agents if agent in self.active_agents]
        
        # If no agents remain after filtering, use all active agents
        if not final_agents:
            final_agents = self.active_agents.copy()
        
        return {
            "mentions_nvidia": mentions_nvidia,
            "is_financial": is_financial,
            "is_historical": is_historical,
            "is_recent": is_recent,
            "mentions_off_topic": mentions_off_topic,
            "recommended_agents": final_agents,
            "original_query": query
        }
    
    def _standardize_date_filters(self, query: str, years: Optional[List[int]] = None, 
                                 quarters: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Extract and standardize date information from the query if not provided explicitly.
        
        Args:
            query: The research question
            years: Optional list of years to filter by
            quarters: Optional list of quarters to filter by
            
        Returns:
            Dict with standardized years and quarters
        """
        # If years and quarters are already provided, just return them
        if years or quarters:
            return {"years": years, "quarters": quarters}
        
        # Otherwise, try to extract date information from the query
        extracted_years = []
        extracted_quarters = []
        
        # Look for years in the query (assuming 4-digit years from 1990 to 2025)
        year_matches = re.findall(r'\b(19[9][0-9]|20[0-2][0-9])\b', query)
        if year_matches:
            extracted_years = [int(year) for year in year_matches]
        
        # Look for quarters in the query
        quarter_matches = re.findall(r'(?:q|Q)([1-4])|(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter', query)
        if quarter_matches:
            for match in quarter_matches:
                if match:
                    if match.isdigit():
                        extracted_quarters.append(int(match))
                    elif match == "first" or match == "1st":
                        extracted_quarters.append(1)
                    elif match == "second" or match == "2nd":
                        extracted_quarters.append(2)
                    elif match == "third" or match == "3rd":
                        extracted_quarters.append(3)
                    elif match == "fourth" or match == "4th":
                        extracted_quarters.append(4)
        
        # If looking for current/recent information but no specific year is mentioned,
        # include the current year (2025)
        if re.search(r'current|recent|latest|this year|now|today', query.lower()) and not extracted_years:
            extracted_years = [2025]
        
        # If looking for "last year" or "previous year" and no specific year is mentioned
        if re.search(r'last year|previous year', query.lower()) and not extracted_years:
            extracted_years = [2024]
        
        return {
            "years": extracted_years if extracted_years else None,
            "quarters": extracted_quarters if extracted_quarters else None
        }
        
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
        
        # Analyze query to determine agents and context
        query_analysis = self._analyze_query_for_agents(query)
        logger.info(f"Query analysis: {query_analysis}")
        
        # Get recommended agents from analysis
        agents_to_use = query_analysis["recommended_agents"]
        logger.info(f"Using agents: {agents_to_use}")
        
        # Standardize date filters
        date_filters = self._standardize_date_filters(query, years, quarters)
        years = date_filters["years"]
        quarters = date_filters["quarters"]
        
        results = {}
        content = {}
        sources = {}
        
        # Track agent failures for fallback handling
        agent_failures = {}
        
        # Process with RAG agent if enabled and recommended
        if "rag" in agents_to_use:
            logger.info("Processing with RAG agent")
            try:
                rag_results = self.rag_agent.query(query, years, quarters)
                results["historical_data"] = {
                    "content": rag_results.get("response", "No historical data available"),
                    "sources": rag_results.get("sources", [])
                }
                content["historical_data"] = rag_results.get("response", "No historical data available")
                sources["historical_data"] = rag_results.get("sources", [])
            except Exception as e:
                error_message = f"Error retrieving historical data: {str(e)}"
                logger.error(f"Error in RAG agent: {str(e)}", exc_info=True)
                results["historical_data"] = {
                    "content": error_message,
                    "sources": []
                }
                content["historical_data"] = error_message
                agent_failures["rag"] = str(e)
        
        # Process with Snowflake agent if enabled and recommended
        if "snowflake" in agents_to_use:
            logger.info("Processing with Snowflake agent")
            try:
                snowflake_results = self.snowflake_agent.query(query, years, quarters)
                results["financial_metrics"] = {
                    "content": snowflake_results.get("response", "No financial metrics available"),
                    "chart": snowflake_results.get("chart", None),
                    "sources": snowflake_results.get("sources", [])
                }
                content["financial_metrics"] = snowflake_results.get("response", "No financial metrics available")
                sources["financial_metrics"] = snowflake_results.get("sources", [])
            except Exception as e:
                error_message = f"Error retrieving financial metrics: {str(e)}"
                logger.error(f"Error in Snowflake agent: {str(e)}", exc_info=True)
                results["financial_metrics"] = {
                    "content": error_message,
                    "chart": None,
                    "sources": []
                }
                content["financial_metrics"] = error_message
                agent_failures["snowflake"] = str(e)
        
        # Process with WebSearch agent if enabled and recommended
        if "websearch" in agents_to_use:
            logger.info("Processing with WebSearch agent")
            try:
                # Pass query analysis context to WebSearch agent
                websearch_results = self.websearch_agent.query(query, years, quarters)
                results["latest_insights"] = {
                    "content": websearch_results.get("response", "No recent insights available"),
                    "sources": websearch_results.get("sources", [])
                }
                content["latest_insights"] = websearch_results.get("response", "No recent insights available")
                sources["latest_insights"] = websearch_results.get("sources", [])
            except Exception as e:
                error_message = f"Error retrieving latest insights: {str(e)}"
                logger.error(f"Error in WebSearch agent: {str(e)}", exc_info=True)
                results["latest_insights"] = {
                    "content": error_message,
                    "sources": []
                }
                content["latest_insights"] = error_message
                agent_failures["websearch"] = str(e)
        
        # Fallback mechanisms if primary agents failed
        if agent_failures:
            logger.warning(f"Some agents failed: {agent_failures}")
            
            # If RAG failed but question seems historical, try WebSearch as fallback
            if "rag" in agent_failures and "websearch" in self.active_agents and "websearch" not in agents_to_use:
                logger.info("Attempting to use WebSearch agent as fallback for RAG")
                try:
                    websearch_results = self.websearch_agent.query(query, years, quarters)
                    results["historical_data_fallback"] = {
                        "content": websearch_results.get("response", "No historical data available from fallback"),
                        "sources": websearch_results.get("sources", [])
                    }
                    # Add to content for synthesis
                    content["historical_data"] = websearch_results.get("response", "No historical data available from fallback")
                    sources["historical_data"] = websearch_results.get("sources", [])
                except Exception as e:
                    logger.error(f"Error in WebSearch fallback: {str(e)}", exc_info=True)
            
            # If Snowflake failed but question is financial, try WebSearch as fallback
            if "snowflake" in agent_failures and "websearch" in self.active_agents and "websearch" not in agents_to_use:
                logger.info("Attempting to use WebSearch agent as fallback for Snowflake")
                try:
                    financial_query = f"NVIDIA financial data {query}"
                    websearch_results = self.websearch_agent.query(financial_query, years, quarters)
                    results["financial_metrics_fallback"] = {
                        "content": websearch_results.get("response", "No financial metrics available from fallback"),
                        "sources": websearch_results.get("sources", [])
                    }
                    # Add to content for synthesis
                    content["financial_metrics"] = websearch_results.get("response", "No financial metrics available from fallback")
                    sources["financial_metrics"] = websearch_results.get("sources", [])
                except Exception as e:
                    logger.error(f"Error in WebSearch fallback: {str(e)}", exc_info=True)
                    
        # Compile all sources for citation in the final report
        all_sources = []
        for source_list in sources.values():
            all_sources.extend(source_list)
            
        # Remove duplicates while maintaining order
        unique_sources = []
        seen_urls = set()
        for source in all_sources:
            normalized_url = source.rstrip('/').lower()
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_sources.append(source)
        
        # Check if this is a non-NVIDIA query that needs special handling
        needs_special_handling = query_analysis["mentions_off_topic"] and not query_analysis["mentions_nvidia"]
        
        # Synthesize the final report
        summary = ""
        
        # For special handling cases, we'll rely on WebSearch agent's response
        if needs_special_handling and "websearch" in agents_to_use and "latest_insights" in content:
            summary = content["latest_insights"]
        elif len(agents_to_use) == 1:
            # If only one agent is used, use its response
            if "rag" in agents_to_use:
                summary = content.get("historical_data", "")
            elif "snowflake" in agents_to_use:
                summary = content.get("financial_metrics", "")
            elif "websearch" in agents_to_use:
                summary = content.get("latest_insights", "")
        else:
            # Otherwise synthesize from all available content
            try:
                # Create system prompt based on query analysis
                system_prompt = self._create_system_prompt(query_analysis)
                
                # Create synthesis prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", """
                    Please create a comprehensive research report answering the following query: {query}
                    
                    Available information:
                    
                    HISTORICAL DATA:
                    {historical_data}
                    
                    FINANCIAL METRICS:
                    {financial_metrics}
                    
                    LATEST INSIGHTS:
                    {latest_insights}
                    
                    Important: Include appropriate numbered citations [1], [2], etc. for all factual information and include 
                    a References section at the end listing full URLs. The minimum length should be 1-2 pages (600-1000 words).
                    """)
                ])
                
                # Generate synthesis
                chain = prompt | self.llm
                response = chain.invoke({
                    "query": query,
                    "historical_data": content.get("historical_data", "Not available"),
                    "financial_metrics": content.get("financial_metrics", "Not available"),
                    "latest_insights": content.get("latest_insights", "Not available"),
                })
                
                summary = response.content
                
            except Exception as e:
                logger.error(f"Error in synthesis: {str(e)}", exc_info=True)
                summary = "Error generating synthesis: " + str(e)
                
                # Fallback to the most relevant content if synthesis fails
                if "websearch" in agents_to_use and "latest_insights" in content:
                    summary = content["latest_insights"]
                elif "rag" in agents_to_use and "historical_data" in content:
                    summary = content["historical_data"]
                elif "snowflake" in agents_to_use and "financial_metrics" in content:
                    summary = content["financial_metrics"]
        
        # Create final report with clear separation between summary and detailed sections
        final_report = {
            "summary": summary,  # Top-level summary (won't be duplicated)
            "details": results,   # Detailed results from each agent
            "sources": unique_sources  # All sources used across agents
        }
        
        return final_report
    
    def _create_system_prompt(self, query_analysis: Dict[str, Any]) -> str:
        """
        Create a tailored system prompt based on query analysis
        
        Args:
            query_analysis: Dictionary with query analysis results
            
        Returns:
            Tailored system prompt for synthesis
        """
        # Base prompt
        base_prompt = """
        You are an elite financial analyst specializing in NVIDIA with exceptional research skills. 
        Your task is to synthesize information from multiple sources to create an EXTENSIVE, 
        data-rich, well-structured report that addresses the research query with extraordinary depth and nuance.
        
        Follow these guidelines for an exceptional report:
        
        1. STRUCTURE: Create a formal report with these comprehensive sections:
           - Executive Summary (concise yet complete overview of key findings and implications)
           - Company Overview (when relevant to the query)
           - Financial Performance Analysis (extensive metrics with YoY and QoQ comparisons, segment breakdowns)
           - Current Market Position (in-depth competitive landscape, market share, strategic advantages)
           - Product Portfolio and Innovation (comprehensive review of technologies, R&D focus, product roadmap)
           - Future Outlook and Projections (detailed growth drivers, challenges, analyst expectations)
           - Conclusion & Investment Implications (synthesis of insights with action-oriented takeaways)
           - References (numbered list of all sources)
        
        2. CONTENT GUIDELINES:
           - Your report MUST be a minimum of 1-2 pages (approximately 600-1000 words)
           - Provide EXACT figures, percentages, and specific metrics wherever possible
           - Include precise year-over-year and quarter-over-quarter growth rates with decimal precision
           - Present comprehensive comparative analysis across multiple time periods
           - Highlight causal relationships between business decisions and financial outcomes
           - Identify emerging patterns and inflection points in NVIDIA's business trajectory
           - Connect historical performance indicators with forward-looking projections
           - Integrate multiple qualitative insights with quantitative data points
           - Include executive statements when relevant to provide strategic context
        
        3. WRITING STYLE:
           - Use authoritative, analytical language that demonstrates deep subject matter expertise
           - Incorporate financial terminology and industry-specific concepts accurately
           - Present information in detailed paragraphs with bulleted lists for key metrics
           - Bold important facts, figures, and key insights for quick reference
           - Maintain objectivity while providing nuanced perspective on implications
           - Use numbered citations [1], [2], etc. throughout the text for all factual claims
        
        4. CITATION AND REFERENCE REQUIREMENTS:
           - For inline citations, use numbered format: [1], [2], etc.
           - Include the citation numbers immediately after the relevant claim
           - At the end of the report, include a "References" section
           - In the References section, list all sources with their full URLs
           - Ensure the citation numbers match between the inline citations and references list
        """
        
        # Add specialized instructions based on query analysis
        if query_analysis["mentions_off_topic"] and not query_analysis["mentions_nvidia"]:
            # Add NVIDIA focus for off-topic queries
            nvidia_focus = """
            IMPORTANT: You are part of the NVIDIA Research Assistant.
            While the query may mention other companies or individuals not directly related to NVIDIA,
            YOU MUST focus your report primarily on NVIDIA's business, performance, and relevance to the query.
            
            - If asked about other tech companies, focus on NVIDIA's relationship with them
            - If asked about individuals not related to NVIDIA, redirect focus to NVIDIA leadership
            - If asked about competitors, focus on NVIDIA's competitive position
            - Always relate information back to NVIDIA's financial performance, market position, or business strategy
            """
            base_prompt += nvidia_focus
        
        # Add financial focus for financial queries
        if query_analysis["is_financial"]:
            financial_focus = """
            FINANCIAL REPORTING REQUIREMENTS:
            - Include precise breakdowns of revenue by business segment:
              * Data Center
              * Gaming
              * Professional Visualization
              * Automotive
            - Present both GAAP and non-GAAP metrics when available
            - Include key financial ratios and their implications
            - Analyze gross margin trends and their drivers
            - Compare NVIDIA's performance to relevant industry benchmarks
            """
            base_prompt += financial_focus
        
        # Add historical context for historical queries
        if query_analysis["is_historical"]:
            historical_focus = """
            HISTORICAL ANALYSIS REQUIREMENTS:
            - Trace the evolution of NVIDIA's business model and strategy
            - Identify key milestones and inflection points in the company's history
            - Connect historical decisions to current business outcomes
            - Analyze leadership decisions in their historical context
            - Provide multi-year trend analysis of key metrics
            """
            base_prompt += historical_focus
        
        # Add recent developments focus for recent queries
        if query_analysis["is_recent"]:
            recent_focus = """
            RECENT DEVELOPMENTS FOCUS:
            - Prioritize the most current financial results and announcements
            - Analyze recent product launches and their market reception
            - Include the latest analyst perspectives and rating changes
            - Highlight recent executive statements and their implications
            - Assess the impact of recent industry trends on NVIDIA
            """
            base_prompt += recent_focus
        
        # Final instructions
        final_instructions = """
        YOUR FINAL REPORT MUST:
        1. Be minimum 600-1000 words in length
        2. Include numbered citations [1], [2], etc. within the text for all factual claims
        3. Have a comprehensive "References" section at the end with full URLs
        4. Provide highly specific, data-rich analysis focused on NVIDIA
        """
        
        return base_prompt + "\n" + final_instructions
