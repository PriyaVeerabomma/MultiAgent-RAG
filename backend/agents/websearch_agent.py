# backend/agents/websearch_agent.py
import os
from typing import Dict, Any, List, Optional
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class WebSearchAgent:
    def __init__(self):
        # Initialize with Tavily API
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")
        
        self.client = TavilyClient(api_key=self.api_key)
        self.llm = ChatOpenAI(temperature=0)
        
    def query(self, query_text: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Query Tavily API for latest information on NVIDIA related to the query.
        
        Args:
            query_text: The query text
            years: Optional list of years to filter by
            quarters: Optional list of quarters to filter by
            
        Returns:
            Dictionary with search results and synthesized information
        """
        # Augment query to focus on NVIDIA and recent information
        augmented_query = f"NVIDIA {query_text}"
        
        # Add year and quarter specifications if provided
        if years and len(years) > 0:
            years_str = " OR ".join([str(year) for year in years])
            augmented_query += f" ({years_str})"
        
        if quarters and len(quarters) > 0:
            # Convert quarters to both numeric and text formats (Q1, first quarter, etc.)
            quarter_terms = []
            for q in quarters:
                quarter_terms.append(f"Q{q}")
                quarter_terms.append(f"q{q}")
                
                # Add text versions for more comprehensive search
                if q == 1:
                    quarter_terms.extend(["first quarter", "1st quarter"])
                elif q == 2:
                    quarter_terms.extend(["second quarter", "2nd quarter"])
                elif q == 3:
                    quarter_terms.extend(["third quarter", "3rd quarter"])
                elif q == 4:
                    quarter_terms.extend(["fourth quarter", "4th quarter"])
            
            quarters_str = " OR ".join(quarter_terms)
            augmented_query += f" ({quarters_str})"
        
        augmented_query += " financial news"
        
        # Log the augmented query
        print(f"Web search augmented query: {augmented_query}")

        # Execute search with Tavily
        response = self.client.search(
            query=augmented_query,
            search_depth="advanced",
            max_results=5,
            include_domains=["forbes.com", "cnbc.com", "bloomberg.com", "reuters.com", "wsj.com"]
        )
        
        # Extract results
        search_results = response.get("results", [])
        
        # Format results
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0),
                "published_date": result.get("published_date", "")
            })
        
        # Generate insights from the search results
        insights = self._generate_insights(formatted_results, query_text, years, quarters)
        
        # Return results
        return {
            "results": formatted_results,
            "response": insights,
            "sources": [result.get("url", "") for result in formatted_results]
        }
        
    def _generate_insights(self, results: List[Dict[str, Any]], query_text: str, 
                         years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> str:
        """Generate insights from search results with awareness of years and quarters"""
        # Format results for the prompt
        context = ""
        for i, result in enumerate(results, 1):
            context += f"{i}. Title: {result['title']}\n"
            context += f"   URL: {result['url']}\n"
            context += f"   Published Date: {result['published_date']}\n"
            # Include only a brief snippet of content to avoid copyright issues
            content_snippet = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            context += f"   Content Snippet: {content_snippet}\n\n"
        
        # Create a filter specification for the prompt
        filter_spec = ""
        if years and len(years) > 0:
            filter_spec += f"Years of interest: {', '.join(map(str, years))}\n"
        if quarters and len(quarters) > 0:
            filter_spec += f"Quarters of interest: {', '.join([f'Q{q}' for q in quarters])}\n"
        
        # Create prompt for insights
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a research analyst specializing in NVIDIA.
            Analyze the following recent news articles about NVIDIA to answer the query.
            Focus on extracting the most relevant and recent insights.
            Provide a balanced perspective considering multiple sources.
            
            IMPORTANT: Do not quote extensively from the articles. Use your own words to summarize insights.
            Include at most one short quote (under 25 words) per source, if necessary.
            
            If specific years or quarters are mentioned in the filter specification, prioritize information relevant to those time periods.
            """),
            ("human", """
            Recent news articles about NVIDIA:
            {context}
            
            {filter_spec}
            
            Query: {query}
            """)
        ])
        
        # Generate insights
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "filter_spec": filter_spec,
            "query": query_text
        })
        
        return response.content