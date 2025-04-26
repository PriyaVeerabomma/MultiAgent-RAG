import os
import re
from typing import Dict, Any, List, Optional, Tuple
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import datetime

class WebSearchAgent:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            print("WARNING: TAVILY_API_KEY environment variable not set!")
            raise ValueError("TAVILY_API_KEY environment variable not set")
        
        print(f"WebSearch agent initialized with Tavily API key: {self.api_key[:5]}...{self.api_key[-4:]}")
        self.client = TavilyClient(api_key=self.api_key)
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.cache = {}  # {cache_key: (timestamp, results)}
        
    def _validate_query(self, query_text: str) -> bool:
        query_lower = query_text.lower()
        nvidia_keywords = [
            "nvidia", "nvda", "jensen huang", "gpu", "ai chip", "semiconductor",
            "financial performance", "market position", "product portfolio", 
            "future outlook", "analyst ratings", "historical data"
        ]
        return any(keyword in query_lower for keyword in nvidia_keywords)

    def _get_cache_key(self, query_text: str, years: Optional[List[int]], quarters: Optional[List[int]], category: str) -> str:
        years_str = ",".join(map(str, years or []))
        quarters_str = ",".join(map(str, quarters or []))
        return f"{query_text}:{years_str}:{quarters_str}:{category}"

    def _filter_results_by_date(self, results: List[Dict[str, Any]], years: Optional[List[int]], quarters: Optional[List[int]]) -> List[Dict[str, Any]]:
        if not years and not quarters:
            return results

        filtered_results = []
        for result in results:
            pub_date = result.get("published_date", "")
            if not pub_date:
                continue

            try:
                pub_date = datetime.datetime.strptime(pub_date, "%Y-%m-%d")
                result_year = pub_date.year
                result_quarter = (pub_date.month - 1) // 3 + 1

                year_match = not years or result_year in years
                quarter_match = not quarters or result_quarter in quarters

                if year_match and quarter_match:
                    filtered_results.append(result)
            except ValueError:
                continue

        return filtered_results

    def query(self, query_text: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> Dict[str, Any]:
        if not self._validate_query(query_text):
            return {
                "results": [],
                "response": "This research assistant is designed for NVIDIA-related queries only. Please ask a question about NVIDIA's financial performance, products, market position, or related topics.",
                "sources": []
            }

        query_context = self._assess_query_context(query_text)
        search_categories = self._generate_search_categories(query_text, query_context)
        relevant_categories = self._select_relevant_categories(search_categories, query_text)

        all_results = []
        current_time = datetime.datetime.now()
        cache_expiry = datetime.timedelta(hours=24)

        for category, category_query in relevant_categories:
            augmented_query = self._build_query(category_query, years, quarters)
            cache_key = self._get_cache_key(augmented_query, years, quarters, category)

            if cache_key in self.cache:
                cached_time, cached_results = self.cache[cache_key]
                if current_time - cached_time < cache_expiry:
                    print(f"Using cached results for category '{category}': {augmented_query}")
                    all_results.extend(cached_results)
                    continue

            print(f"Running search for category '{category}': {augmented_query}")
            try:
                response = self.client.search(
                    query=augmented_query,
                    search_depth="advanced",
                    max_results=3,
                    include_domains=[
                        "nvidia.com", "investor.nvidia.com", "nvidianews.nvidia.com",
                        "finance.yahoo.com", "reuters.com", "bloomberg.com", "wsj.com",
                        "investopedia.com", "seekingalpha.com", "marketwatch.com", "fool.com", "morningstar.com",
                        "forbes.com", "cnbc.com", "macrotrends.net"
                    ]
                )
                search_results = response.get("results", [])
                category_results = []
                for result in search_results:
                    category_results.append({
                        "category": category,
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "score": result.get("score", 0),
                        "published_date": result.get("published_date", "")
                    })
                self.cache[cache_key] = (current_time, category_results)
                all_results.extend(category_results)
            except Exception as e:
                print(f"Error searching for category '{category}': {str(e)}")

        all_results = self._filter_results_by_date(all_results, years, quarters)
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        unique_results = []
        seen_urls = set()
        for result in all_results:
            url = result.get("url", "")
            normalized_url = url.lower().rstrip('/')
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_results.append(result)

        if query_context["needs_nvidia_focus"]:
            filtered_results = self._filter_results_for_nvidia(unique_results)
            top_results = filtered_results[:15]
        else:
            top_results = unique_results[:15]

        insights = self._generate_detailed_report(top_results, query_text, query_context, years, quarters)
        return {
            "results": top_results,
            "response": insights,
            "sources": [result.get("url", "") for result in top_results]
        }
    
    def _assess_query_context(self, query_text: str) -> Dict[str, Any]:
        query_lower = query_text.lower()
        mentions_nvidia = any(term in query_lower for term in ["nvidia", "nvda", "jensen huang"])
        is_financial = any(term in query_lower for term in [
            "financial", "revenue", "earnings", "profit", "income", "stock", 
            "share", "market", "growth", "performance", "quarter", "fiscal"
        ])
        mentions_other_entities = any(term in query_lower for term in [
            "elon musk", "tesla", "spacex", "amazon", "apple", "microsoft", 
            "meta", "facebook", "intel", "amd", "arm", "qualcomm"
        ])
        return {
            "mentions_nvidia": mentions_nvidia,
            "is_financial": is_financial,
            "mentions_other_entities": mentions_other_entities,
            "needs_nvidia_focus": mentions_other_entities and not mentions_nvidia,
            "original_query": query_text
        }
    
    def _filter_results_for_nvidia(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        nvidia_related = []
        for result in results:
            content = result.get("content", "").lower()
            title = result.get("title", "").lower()
            if ("nvidia" in content or "nvda" in content or "nvidia" in title or "nvda" in title):
                nvidia_related.append(result)
        if len(nvidia_related) >= 5:
            return nvidia_related
        return nvidia_related + [r for r in results if r not in nvidia_related][:15-len(nvidia_related)]
    
    def _select_relevant_categories(self, categories: List[Tuple[str, str]], query: str) -> List[Tuple[str, str]]:
        query_lower = query.lower()
        category_scores = {
            "Overview": 3, "Financial": 5, "Products": 2, "Market": 4, "Outlook": 3, "Recent": 4,
            "Executive": 2, "Analyst": 4, "Metrics": 5, "Industry": 3, "Investor": 4, "Historical": 2
        }
        if any(term in query_lower for term in ["revenue", "earning", "profit", "income", "eps"]):
            category_scores["Financial"] += 3
            category_scores["Metrics"] += 3
            category_scores["Investor"] += 2
        if any(term in query_lower for term in ["product", "gpu", "chip", "hardware", "ai"]):
            category_scores["Products"] += 3
            category_scores["Industry"] += 2
        if any(term in query_lower for term in ["market", "competition", "share", "position"]):
            category_scores["Market"] += 3
            category_scores["Industry"] += 2
        if any(term in query_lower for term in ["future", "forecast", "outlook", "prediction"]):
            category_scores["Outlook"] += 3
            category_scores["Analyst"] += 2
        if any(term in query_lower for term in ["recent", "latest", "new", "current"]):
            category_scores["Recent"] += 3
        if any(term in query_lower for term in ["ceo", "executive", "jensen", "leadership"]):
            category_scores["Executive"] += 3
        if any(term in query_lower for term in ["history", "past", "founded", "beginning"]):
            category_scores["Historical"] += 3
            category_scores["Overview"] += 1
        scored_categories = [(cat, query_text, category_scores.get(cat, 0)) 
                             for cat, query_text in categories]
        scored_categories.sort(key=lambda x: x[2], reverse=True)
        return [(cat, query) for cat, query, score in scored_categories[:5]]
    
    def _generate_search_categories(self, query_text: str, query_context: Dict[str, Any]) -> List[Tuple[str, str]]:
        categories = []
        if query_context["needs_nvidia_focus"]:
            query_prefix = "NVIDIA "
        else:
            query_prefix = ""
        categories.extend([
            ("Overview", f"{query_prefix}company overview {query_text}"),
            ("Financial", f"{query_prefix}financial performance revenue earnings {query_text}"),
            ("Products", f"{query_prefix}products GPU AI chips {query_text}"),
            ("Market", f"{query_prefix}market position competition market share {query_text}"),
            ("Outlook", f"{query_prefix}future outlook forecast analysts prediction {query_text}"),
            ("Recent", f"{query_prefix}recent news developments this month {query_text}"),
        ])
        if query_context["is_financial"]:
            categories.extend([
                ("Metrics", f"{query_prefix}financial metrics EPS revenue growth {query_text}"),
                ("Investor", f"{query_prefix}investor relations shareholder quarterly report {query_text}"),
                ("Analyst", f"{query_prefix}analyst reports ratings consensus {query_text}"),
            ])
        categories.append(("Industry", f"{query_prefix}semiconductor industry comparison AMD Intel {query_text}"))
        categories.append(("Executive", f"{query_prefix}Jensen Huang CEO statements {query_text}"))
        categories.append(("Historical", f"{query_prefix}history historical data performance {query_text}"))
        return categories
        
    def _build_query(self, category_query: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> str:
        augmented_query = category_query
        if years and len(years) > 0:
            years_str = " OR ".join([str(year) for year in years])
            augmented_query += f" ({years_str})"
        if quarters and len(quarters) > 0:
            quarter_terms = []
            for q in quarters:
                quarter_terms.append(f"Q{q}")
                quarter_terms.append(f"q{q}")
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
        return augmented_query
        
    def _generate_detailed_report(self, results: List[Dict[str, Any]], query_text: str, query_context: Dict[str, Any],
                                  years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> str:
        categorized_results = {}
        for result in results:
            category = result.get("category", "General")
            if category not in categorized_results:
                categorized_results[category] = []
            categorized_results[category].append(result)
        
        context = ""
        for category, category_results in categorized_results.items():
            context += f"## {category} Information:\n\n"
            for i, result in enumerate(category_results, 1):
                context += f"{i}. Title: {result['title']}\n"
                context += f"   Source: {result['url']}\n"
                context += f"   Published Date: {result['published_date']}\n"
                content_snippet = result['content'][:350] + "..." if len(result['content']) > 350 else result['content']
                context += f"   Content: {content_snippet}\n\n"
        
        filter_desc = ""
        if years and len(years) > 0:
            filter_desc += f"Years of interest: {', '.join(map(str, years))}\n"
        if quarters and len(quarters) > 0:
            filter_desc += f"Quarters of interest: {', '.join([f'Q{q}' for q in quarters])}\n"
        
        sources = []
        for result in results:
            sources.append({
                "url": result.get("url", ""),
                "title": result.get("title", "")
            })
        sources_str = "\n".join([f"{i+1}. {source['url']}" for i, source in enumerate(sources)])
        
        system_prompt = self._create_system_prompt(query_context)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
            Here are the research results about {original_query}:
            
            {context}
            
            {filter_spec}
            
            Primary Query: {query}
            
            Please create a detailed, comprehensive research report based on this information.
            Include appropriate numbered citations [1], [2], etc. WITHIN the text where you make factual claims,
            and include a numbered References section at the end with full URLs.
            
            Sources to reference:
            {sources}
            """)
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "filter_spec": filter_desc,
            "query": query_text,
            "original_query": query_context["original_query"],
            "sources": sources_str
        })
        return response.content
    
    def _create_system_prompt(self, query_context: Dict[str, Any]) -> str:
        base_prompt = """
        You are a professional financial analyst specializing in the semiconductor industry with expertise in NVIDIA.
        Your task is to create a detailed, data-rich research report based on the provided search results.
        
        Follow these guidelines for an exceptional report:
        
        1. LENGTH REQUIREMENT:
           - Your report MUST be a minimum of 1-2 pages (approximately 600-1000 words)
           - Ensure comprehensive coverage of ALL relevant aspects of the query
           - Include detailed analysis in EACH section, not just summary statements
        
        2. STRUCTURE: Create a formal report with these comprehensive sections:
           - Executive Summary (thorough overview of key findings)
           - Company Overview (when relevant to the query)
           - Financial Performance (extensive metrics with precise figures by segment)
           - Market Position (detailed competitive landscape with exact market share figures)
           - Product Portfolio (comprehensive analysis of technologies driving revenue)
           - Future Outlook (detailed analyst predictions with specific growth estimates)
           - Conclusion & Investment Implications (actionable insights)
           - References (numbered list of all sources used)
        
        3. CONTENT GUIDELINES:
           - Provide EXACT numbers, percentages, and growth rates wherever available
           - Include detailed year-over-year and quarter-over-quarter comparisons with precise figures
           - Present COMPREHENSIVE metrics like revenue by segment, profit margins, EPS, etc.
           - Cite specific analyst projections with firm names when available
           - Incorporate direct quotes from company executives (limited to under 25 words per quote)
           - Present balanced perspectives with both bullish and bearish viewpoints
           - Analyze implications of data with depth and sophistication
           - Provide segment-by-segment breakdowns when data is available
           - Include historical context to frame current performance
        
        4. WRITING STYLE:
           - Use authoritative, technical language appropriate for financial analysts
           - Employ precise financial terminology and industry-specific concepts
           - Present information in detailed paragraphs with substantive data points
           - Use bulleted lists for key metrics and significant data points
           - Bold important figures, growth rates, and critical insights for quick reference
           - Maintain analytical objectivity while providing expert interpretation
        
        5. CITATION AND REFERENCE REQUIREMENTS:
           - Use numbered citations [1], [2], etc. within the text for all factual claims
           - Place the citation number immediately after the relevant claim or data point
           - Include a "References" section at the end with full URLs
           - The numbered citations should correspond to the numbered references
           - Each source should be properly attributed in the References section
           - Format:
              - Within text: "NVIDIA reported quarterly revenue of $39.3 billion [1]..."
              - In References: "1. https://investor.nvidia.com/financial-info/..."
        
        6. TIME RELEVANCE:
           - STRICTLY adhere to the specified years and quarters in the filter specification
           - If no years or quarters are specified, focus on the most recent data available
           - Clearly state the time period for all data points (e.g., "Q3 2023 revenue")
        """
        if query_context["needs_nvidia_focus"]:
            nvidia_focus = """
            IMPORTANT: You are part of the NVIDIA Research Assistant. 
            While the query may mention other companies or individuals, your report MUST focus primarily on NVIDIA.
            
            - If asked about other tech companies, focus on NVIDIA's relationship with them
            - If asked about individuals not related to NVIDIA, redirect to NVIDIA's leadership (e.g., Jensen Huang)
            - If asked about competitive products, focus on NVIDIA's competing products
            - Always relate information back to NVIDIA's financial performance, market position, or business strategy
            
            Your mandate is to provide NVIDIA-focused research, not general information about unrelated topics.
            """
            base_prompt += nvidia_focus
        if query_context["is_financial"]:
            financial_focus = """
            FINANCIAL REPORTING REQUIREMENTS:
            - Include precise financial metrics with exact figures
            - Present year-over-year and quarter-over-quarter growth rates
            - Break down revenue by business segments (Data Center, Gaming, Professional Visualization, Auto)
            - Include both GAAP and non-GAAP metrics when available
            - Provide context for financial performance relative to industry averages and competitors
            - Analyze key financial ratios and their implications for investors
            """
            base_prompt += financial_focus
        base_prompt += """
        YOUR FINAL REPORT MUST:
        1. Be minimum 600-1000 words
        2. Include numbered citations [1], [2], etc. within the text
        3. Have a "References" section at the end with full URLs
        4. Provide highly specific, data-rich analysis
        """
        return base_prompt