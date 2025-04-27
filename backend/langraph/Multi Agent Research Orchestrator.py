from typing import Dict, Any, List, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import logging
import time
from dotenv import load_dotenv

# Import specialized report template
from langraph.report_template import create_research_report_prompt

# Import agent modules
from agents.rag_agent import RagAgent
from agents.snowflake_agent import SnowflakeAgent
from agents.websearch_agent import WebSearchAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAgentResearchOrchestrator:
    """
    Central orchestrator for multi-agent research execution and synthesis.
    Features:
    - Modular agent management
    - Source tracking and error handling
    - Synthesis of structured research reports
    - Optimized for scalability and maintainability
    """

    def __init__(
        self,
        use_rag: bool = True,
        use_snowflake: bool = True,
        use_websearch: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the orchestrator with selected agents.
        """
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Load environment variables
        self._load_environment()

        # Initialize the core LLM for synthesis
        self.llm = ChatOpenAI(
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-0125-preview",
            max_tokens=4000
        )

        # Activate agents
        self.rag_agent = RagAgent(verbose=verbose) if use_rag else None
        self.snowflake_agent = SnowflakeAgent() if use_snowflake else None
        self.websearch_agent = WebSearchAgent() if use_websearch else None

        # Keep track of active agents
        self.active_agents = [
            name for name, agent in zip(
                ["rag", "snowflake", "websearch"],
                [self.rag_agent, self.snowflake_agent, self.websearch_agent]
            ) if agent is not None
        ]

        logger.info(f"Initialized MultiAgentResearchOrchestrator with agents: {self.active_agents}")

    def _load_environment(self):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root_dir = os.path.dirname(backend_dir)
        env_files = [
            os.path.join(root_dir, '.env'),
            os.path.join(backend_dir, '.env')
        ]

        for env_file in env_files:
            if os.path.exists(env_file):
                load_dotenv(env_file)
                print(f"Loaded environment variables from: {env_file}")
                return

        load_dotenv()  # Default fallback
        logger.warning("Using default environment .env loading")

    def run(self, query: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run the orchestrator to gather and synthesize research findings.
        """
        start_time = time.time()
        logger.info(f"Starting research orchestration for query: {query}")

        results, content = {}, {}

        # --- Agent Queries --- #

        if self.rag_agent:
            results["historical_data"], content["historical_data"] = self._query_agent(
                self.rag_agent, "Historical Data", query, years, quarters
            )

        if self.snowflake_agent:
            results["financial_metrics"], content["financial_metrics"] = self._query_agent(
                self.snowflake_agent, "Financial Metrics", query, years, quarters
            )

        if self.websearch_agent:
            results["latest_insights"], content["latest_insights"] = self._query_agent(
                self.websearch_agent, "Latest Insights", query, years, quarters
            )

        # --- Report Synthesis --- #

        if len(self.active_agents) == 1:
            # If only one active agent, return its output directly
            final_report = list(content.values())[0]
        else:
            final_report = self._synthesize_report(query, content)

        total_time = time.time() - start_time
        logger.info(f"Research orchestration completed in {total_time:.2f}s")

        return {
            "content": final_report,
            **results,
            "processing_time": f"{total_time:.2f}s"
        }

    def _query_agent(self, agent, agent_name: str, query: str, years: Optional[List[int]], quarters: Optional[List[int]]):
        logger.info(f"Querying {agent_name} agent")
        try:
            result = agent.query(query, years, quarters)
            return (
                {
                    "content": result.get("response", f"No {agent_name} available."),
                    "sources": result.get("sources", []),
                    "confidence_score": result.get("confidence_score", None)
                },
                result.get("response", f"No {agent_name} available.")
            )
        except Exception as e:
            logger.error(f"Error querying {agent_name}: {str(e)}", exc_info=True)
            return (
                {
                    "content": f"Error retrieving {agent_name}: {str(e)}",
                    "sources": [],
                    "confidence_score": 0
                },
                f"Error retrieving {agent_name}: {str(e)}"
            )

    def _synthesize_report(self, query: str, content: Dict[str, str]) -> str:
        """
        Synthesize a final report from the agent outputs.
        """
        try:
            logger.info("Synthesizing final report")
            prompt = create_research_report_prompt()
            chain = prompt | self.llm
            response = chain.invoke({
                "query": query,
                "historical_data": content.get("historical_data", "Not available."),
                "financial_metrics": content.get("financial_metrics", "Not available."),
                "latest_insights": content.get("latest_insights", "Not available.")
            })
            return response.content
        except Exception as e:
            logger.error(f"Error synthesizing final report: {str(e)}", exc_info=True)
            return "Error generating final synthesis: " + str(e)
