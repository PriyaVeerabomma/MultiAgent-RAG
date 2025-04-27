import os
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchAgent:
    def __init__(self):
        # Load environment variables from root first, then backend
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root_dir = os.path.dirname(backend_dir)
        
        # Check if .env exists in root directory first
        root_env_file = os.path.join(root_dir, '.env')
        backend_env_file = os.path.join(backend_dir, '.env')
        
        if os.path.exists(root_env_file):
            load_dotenv(root_env_file)
            print(f"Using .env file from root directory: {root_env_file}")
        elif os.path.exists(backend_env_file):
            load_dotenv(backend_env_file)
            print(f"Using .env file from backend directory: {backend_env_file}")
        else:
            load_dotenv()  # Fall back to default behavior
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.error("TAVILY_API_KEY environment variable not set!")
            raise ValueError("TAVILY_API_KEY environment variable not set")
        
        logger.info(f"WebSearch agent initialized with Tavily API key: {self.api_key[:5]}...{self.api_key[-4:]}")
        
        # Initialize Tavily client and LLM
        self.client = TavilyClient(api_key=self.api_key)
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        
        # Cache to store results and avoid repetitive searches
        self.cache = {}
        
        # Trusted domains for NVIDIA research
        self.trusted_domains = [
            "nvidia.com", "investor.nvidia.com", "finance.yahoo.com", 
            "reuters.com", "bloomberg.com", "wsj.com", "cnbc.com", 
            "techcrunch.com", "theverge.com", "wired.com", "seekingalpha.com"
        ]
    
    def query(self, query_text: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> Dict[str, Any]:
        """Search for NVIDIA information based on user query and time filters"""
        logger.info(f"Web search for: '{query_text}', Years: {years}, Quarters: {quarters}")
        
        # Step 1: Generate appropriate search queries
        search_queries = self._generate_search_queries(query_text, years, quarters)
        
        # Step 2: Execute searches and collect results
        all_results = []
        for query in search_queries:
            try:
                results = self._search(query)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Search error for '{query}': {str(e)}")
        
        # Step 3: Apply light filtering for years/quarters if specified
        if years or quarters:
            all_results = self._filter_by_time(all_results, years, quarters)
            
            # If multi-year query with no results, try again without filtering
            if not all_results and years and len(years) > 2:
                logger.info("Multi-year query with no results after filtering, using unfiltered results")
                # Try a simpler query for broad multi-year overview
                overview_query = f"NVIDIA {query_text} {min(years)}-{max(years)}"
                try:
                    all_results = self._search(overview_query)
                except Exception as e:
                    logger.error(f"Overview search error: {str(e)}")
        
        # Step 4: Process and deduplicate results
        unique_results = self._process_results(all_results)
        
        # Step 5: Generate response or handle no results case
        if not unique_results:
            return {
                "response": self._generate_no_results_message(query_text, years, quarters),
                "results": [],
                "sources": []
            }
        
        # Generate report from search results
        report = self._generate_report(unique_results, query_text, years, quarters)
        
        return {
            "response": report,
            "results": unique_results[:10],  # Limit to top 10 for response
            "sources": [r.get("url", "") for r in unique_results[:10]]
        }
    
    def _generate_search_queries(self, query: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> List[str]:
        """Generate appropriate search queries based on user query and time filters"""
        # Base query always includes NVIDIA
        base = f"NVIDIA {query}"
        queries = [base]  # Always include the basic query
        
        # Handle time filters
        if years:
            if len(years) == 1:
                # Single year
                year = years[0]
                queries.append(f"{base} {year}")
                
                # Add quarter if specified
                if quarters and len(quarters) == 1:
                    quarter = quarters[0]
                    queries.append(f"{base} Q{quarter} {year}")
            elif len(years) > 1:
                # Multiple years - add range format
                year_range = f"{min(years)}-{max(years)}"
                queries.append(f"{base} {year_range}")
                # Also add "historical" or "over time" for multi-year
                queries.append(f"{base} historical {min(years)} to {max(years)}")
        elif quarters:
            # Quarters without specific years
            for q in quarters:
                queries.append(f"{base} Q{q}")
        else:
            # No time filter - add "latest" and "recent"
            queries.append(f"{base} latest")
            queries.append(f"{base} recent")
        
        # Add query variants based on topic
        lower_query = query.lower()
        if any(term in lower_query for term in ["stock", "price", "market", "share"]):
            queries.append(f"NVIDIA stock performance {' '.join([str(y) for y in years]) if years else 'recent'}")
        
        if any(term in lower_query for term in ["innovation", "product", "technology", "launch"]):
            queries.append(f"NVIDIA product innovations {' '.join([str(y) for y in years]) if years else 'recent'}")
        
        if any(term in lower_query for term in ["revenue", "earnings", "financial", "profit"]):
            queries.append(f"NVIDIA financial results {' '.join([str(y) for y in years]) if years else 'recent'}")
        
        # Ensure we don't have too many queries (for efficiency)
        return queries[:5]
    
    def _search(self, query: str) -> List[Dict[str, Any]]:
        """Execute search and return results"""
        # Check cache
        if query in self.cache:
            logger.info(f"Using cached results for: {query}")
            return self.cache[query]
        
        try:
            # First try with trusted domains
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_domains=self.trusted_domains
            )
            
            results = response.get("results", [])
            
            # If no results from trusted domains, try without domain restriction
            if not results:
                response = self.client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=5
                )
                results = response.get("results", [])
            
            # Process results
            processed_results = []
            for result in results:
                processed_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "published_date": result.get("published_date", ""),
                    "score": result.get("score", 0)
                })
            
            # Cache results
            self.cache[query] = processed_results
            logger.info(f"Search for '{query}' returned {len(processed_results)} results")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
    
    def _filter_by_time(self, results: List[Dict[str, Any]], years: Optional[List[int]], quarters: Optional[List[int]]) -> List[Dict[str, Any]]:
        """Simple time-based filtering of results"""
        if not years and not quarters:
            return results
        
        filtered = []
        
        # For multi-year queries (3+ years), be very lenient
        is_multi_year = years and len(years) > 2
        
        for result in results:
            content = result.get("content", "").lower() + " " + result.get("title", "").lower()
            
            # Check publication date if available
            pub_date = result.get("published_date", "")
            date_match = False
            
            if pub_date:
                try:
                    date = datetime.datetime.strptime(pub_date, "%Y-%m-%d")
                    if (not years or date.year in years) and (not quarters or (date.month-1)//3+1 in quarters):
                        date_match = True
                except:
                    pass
            
            # If date doesn't match, check content
            content_match = False
            if not date_match:
                # For multi-year queries, just need to mention ANY of the years
                if is_multi_year:
                    if not years or any(str(year) in content for year in years):
                        content_match = True
                # For specific year/quarter, be more strict
                else:
                    year_match = not years or any(re.search(f"\\b{year}\\b", content) for year in years)
                    quarter_match = not quarters
                    
                    if quarters:
                        for q in quarters:
                            if f"q{q}" in content or f"{self._quarter_name(q)} quarter" in content:
                                quarter_match = True
                                break
                    
                    content_match = year_match and quarter_match
            
            if date_match or content_match:
                filtered.append(result)
        
        # For multi-year queries, if no results match, return all results
        # (as they might discuss the topic without specific year mentions)
        if is_multi_year and not filtered and results:
            return results
            
        return filtered
    
    def _process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and deduplicate search results"""
        # Remove duplicates
        seen_urls = set()
        unique = []
        
        for result in results:
            url = result.get("url", "").lower().rstrip("/")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(result)
        
        # Sort by relevance score
        return sorted(unique, key=lambda x: x.get("score", 0), reverse=True)
    
    def _generate_report(self, results: List[Dict[str, Any]], query: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> str:
        """Generate a comprehensive, in-depth research report from search results"""
        # Format results for the prompt
        results_text = ""
        for i, result in enumerate(results[:10], 1):  # Limit to top 10
            results_text += f"SOURCE {i}:\n"
            results_text += f"Title: {result['title']}\n"
            results_text += f"URL: {result['url']}\n"
            
            if result.get("published_date"):
                results_text += f"Date: {result['published_date']}\n"
            
            # Truncate long content
            content = result.get("content", "")
            if len(content) > 800:
                content = content[:800] + "...[truncated]"
            results_text += f"Content: {content}\n\n"
        
        # Format time period for prompt
        time_period = ""
        if years:
            time_period += f"Years: {', '.join(map(str, years))}"
            if quarters:
                time_period += f", Quarters: {', '.join([f'Q{q}' for q in quarters])}"
        elif quarters:
            time_period += f"Quarters: {', '.join([f'Q{q}' for q in quarters])}"
        else:
            time_period = "Recent information"
        
        # Create prompt for comprehensive analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are NVIDIA Market Intelligence Director, an elite financial and technology analyst with 25+ years of experience 
            analyzing NVIDIA and the semiconductor industry. Your research reports are renowned for their exceptional depth, 
            precision, and actionable insights.
            
            MISSION:
            Create an authoritative, comprehensive research report based on the search results provided. Your report 
            should deliver an unparalleled level of analysis that would satisfy professional investors, corporate executives, 
            and industry analysts seeking deep understanding of NVIDIA.
            
            REPORT REQUIREMENTS:
            
            1. EXTRAORDINARY DEPTH & PERSPECTIVE:
               - Provide 800-1000 words of sophisticated, nuanced analysis
               - Integrate multiple angles and viewpoints into a cohesive narrative
               - Place information in proper industry, market, and technological context
               - Deliver insights that go beyond surface-level observations
               - Connect data points across sources to identify patterns and implications
            
            2. PRECISION & DETAIL:
               - Use exact figures, percentages, and metrics with proper notation
               - Specify exact dates, timeframes, and periods for all references
               - Include comprehensive market data (P/E ratios, market cap, growth rates)
               - Differentiate between announced plans, current projects, and completed initiatives
               - Provide detailed product specifications and performance benchmarks when relevant
            
            3. BALANCED PERSPECTIVE:
               - Present both bullish and bearish viewpoints with supporting evidence
               - Analyze strengths alongside vulnerabilities
               - Compare NVIDIA's position to key competitors with specific metrics
               - Acknowledge limitations in available information
               - Consider both technical and fundamental factors
            
            COMPREHENSIVE STRUCTURE:
            
            1. EXECUTIVE BRIEF (2-3 paragraphs):
               - Synthesize key findings with exceptional clarity
               - Highlight most significant metrics and developments
               - Establish the central narrative and importance
            
            2. MARKET & FINANCIAL POSITION (2-3 paragraphs):
               - Detailed stock performance analysis with specific metrics
               - Market share data with exact percentages across segments
               - Revenue and earnings figures with YoY and QoQ comparisons
               - Valuation metrics compared to industry standards
               - Analyst consensus and target price evaluations
            
            3. TECHNOLOGICAL & PRODUCT ANALYSIS (2-3 paragraphs):
               - In-depth assessment of product portfolio and development pipeline
               - Technical specifications and performance benchmarks
               - R&D investments and innovation trajectory
               - Technology advantages and challenges vs. competitors
               - Product roadmap and strategic technology positioning
            
            4. COMPETITIVE LANDSCAPE (1-2 paragraphs):
               - Detailed competitive position analysis by segment
               - Specific threat and opportunity assessment
               - Market dynamics and shifting competitive advantages
               - Strategic partnerships and ecosystem development
               - Regulatory and market access considerations
            
            5. FORWARD OUTLOOK (1-2 paragraphs):
               - Growth catalysts and potential headwinds
               - Upcoming product launches and expected impact
               - Market expansion opportunities
               - Key risks and mitigating factors
               - Long-term strategic positioning
            
            6. COMPREHENSIVE CONCLUSION (1 paragraph):
               - Synthesize key insights into actionable intelligence
               - Present balanced assessment of prospects
               - Highlight critical factors to monitor
            
            7. REFERENCES:
               - Numbered citation format [1], [2], etc.
               - Complete source listing with accessible URLs
               
            PROFESSIONAL REQUIREMENTS:
            1. CITATION STANDARDS: Use numbered citations [1], [2] etc. for EVERY factual claim
            2. EXACT QUOTES: If directly quoting a source, use quotation marks and keep quotes under 25 words
            3. COPYRIGHT COMPLIANCE: Never reproduce large sections of source material; synthesize and analyze instead
            4. FORMATTING: Use professional formatting with clear section headings, bold for key points, and bullet lists where appropriate
            
            YOUR REPORT MUST BE: 
            - Exceptionally detailed and comprehensive
            - Built exclusively on information from the provided sources
            - Properly cited throughout
            - Structured with clear section headings
            - Written at the level of a professional industry analyst report
            """),
            ("human", f"""
            RESEARCH QUESTION: "{query}"
            
            TIME PERIOD FOCUS: {time_period}
            
            SOURCE MATERIALS:
            {results_text}
            
            Please produce a comprehensive, authoritative research report addressing the question based ONLY on the information 
            provided in these sources. Use numbered citations [1], [2], etc. for every factual claim.
            
            If the available sources have limitations in answering certain aspects of the question, acknowledge these gaps 
            while providing the most thorough analysis possible with the available information.
            
            Your report should be exceptionally detailed and insightful - the kind of analysis that would be valuable to 
            professional investors and industry analysts seeking deep understanding of NVIDIA.
            """)
        ])
        
        # Generate report
        chain = prompt | self.llm
        response = chain.invoke({})
        
        return response.content
    
    def _generate_no_results_message(self, query: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> str:
        """Generate a helpful message when no results are found"""
        time_desc = ""
        if years:
            time_desc += f"years {', '.join(map(str, years))}"
            if quarters:
                time_desc += f" and quarters {', '.join([f'Q{q}' for q in quarters])}"
        elif quarters:
            time_desc += f"quarters {', '.join([f'Q{q}' for q in quarters])}"
        
        if time_desc:
            return f"I couldn't find specific information about NVIDIA related to '{query}' during {time_desc}. This may be because this information isn't readily available online or because the search didn't return relevant results for this specific time period. You could try broadening your search by using fewer time filters or reformulating your question."
        else:
            return f"I couldn't find specific information about NVIDIA related to '{query}'. This may be because this information isn't readily available online or because the search didn't return relevant results. You could try reformulating your question to be more specific or to use different keywords."
    
    def _quarter_name(self, quarter: int) -> str:
        """Convert quarter number to name"""
        names = {1: "first", 2: "second", 3: "third", 4: "fourth"}
        return names.get(quarter, "")