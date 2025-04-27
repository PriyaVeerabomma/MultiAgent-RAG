import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import snowflake.connector
import base64
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnowflakeAgent:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get Snowflake credentials
        self.snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.snowflake_user = os.getenv("SNOWFLAKE_USER")
        self.snowflake_password = os.getenv("SNOWFLAKE_PASSWORD")
        self.snowflake_database = os.getenv("SNOWFLAKE_DATABASE", "NVIDIA_DB")
        self.snowflake_schema = os.getenv("SNOWFLAKE_SCHEMA", "STOCK_DATA")
        self.snowflake_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        
        logger.info(f"Initializing Snowflake connection with account: {self.snowflake_account}, user: {self.snowflake_user}")
        logger.info(f"Database: {self.snowflake_database}, Schema: {self.snowflake_schema}")
        
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0)
        
        # CSV file path - use environment variable with fallback to local path
        csv_path_env = os.getenv("NVDA_CSV_PATH")
        if csv_path_env:
            self.csv_path = csv_path_env
        else:
            # Default to local path
            self.csv_path = os.path.join(os.path.dirname(__file__), "..", "NVDA_5yr_history_20250407.csv")
        
        # Create charts directory - use environment variable or default
        charts_dir_env = os.getenv("CHARTS_DIR")
        if charts_dir_env:
            self.charts_dir = charts_dir_env
        else:
            self.charts_dir = os.path.join(os.path.dirname(__file__), "..", "charts")
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Test connection to Snowflake
        try:
            self._test_connection()
            logger.info("Successfully connected to Snowflake")
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
    
    def _test_connection(self):
        """Test the connection to Snowflake"""
        conn = self.connect_to_snowflake()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT current_version()")
                version = cursor.fetchone()[0]
                logger.info(f"Connected to Snowflake version: {version}")
        finally:
            conn.close()
    
    def connect_to_snowflake(self):
        """Create a connection to Snowflake"""
        return snowflake.connector.connect(
            account=self.snowflake_account,
            user=self.snowflake_user,
            password=self.snowflake_password,
            database=self.snowflake_database,
            schema=self.snowflake_schema,
            warehouse=self.snowflake_warehouse
        )
    
    def query(self, query_text: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Query NVIDIA financial data based on query text, years, and quarters.
        Uses SQL to query the Snowflake database with LLM-generated queries.
        """
        try:
            # Generate SQL query using LLM based on user query and filters
            sql_query = self._generate_sql_with_llm(query_text, years, quarters)
            logger.info(f"LLM-generated SQL query: {sql_query}")
            
            # Execute the query
            try:
                df = self._execute_query(sql_query)
                logger.info(f"Query executed, retrieved {len(df)} rows")
            except Exception as query_error:
                logger.error(f"Error executing LLM-generated query: {str(query_error)}")
                # Fall back to a simpler query
                fallback_query = self._generate_fallback_query(years, quarters)
                logger.info(f"Using fallback query: {fallback_query}")
                df = self._execute_query(fallback_query)
                logger.info(f"Fallback query executed, retrieved {len(df)} rows")
            
            if df is None or df.empty:
                # Check if the query is about revenue which we don't have in our stock data
                if "revenue" in query_text.lower() or "earnings" in query_text.lower():
                    return {
                        "response": f"I don't have revenue or earnings data in the stock database. The available data includes stock prices (open, close, high, low), trading volume, moving averages (50-day, 200-day), and returns metrics (daily, monthly, yearly). I can analyze these stock metrics for the time periods you specified, but cannot provide direct revenue information.",
                        "chart": None,
                        "sources": ["Snowflake - NVIDIA_DB.STOCK_DATA.NVDA_STOCK_DATA"]
                    }
                else:
                    return {
                        "response": f"No stock data found for NVIDIA with the specified filters (Years: {years}, Quarters: {quarters}). Please try different time periods or a broader query.",
                        "chart": None,
                        "sources": ["Snowflake - NVIDIA_DB.STOCK_DATA.NVDA_STOCK_DATA"]
                    }
            
            # Generate chart if data is available
            chart_path = None
            if not df.empty:
                chart_metric = self._determine_best_chart_metric(df, query_text)
                chart_path = self._generate_chart(df, chart_metric)
            
            # Generate analysis with LLM
            analysis = self._generate_analysis(df, query_text, years, quarters, sql_query)
            
            # If the query is about revenue but we only have stock data
            if "revenue" in query_text.lower() or "earnings" in query_text.lower():
                analysis += "\n\n*Note: This analysis is based on NVIDIA's stock performance data, not actual revenue figures. Stock performance can be an indicator of financial health but doesn't directly represent revenue.*"
            
            return {
                "response": analysis,
                "chart": chart_path,
                "sources": ["Snowflake - NVIDIA_DB.STOCK_DATA.NVDA_STOCK_DATA"]
            }
            
        except Exception as e:
            import traceback
            error_message = f"Error processing data: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_message)
            
            # Try to use CSV as fallback
            try:
                logger.info(f"Attempting to use CSV fallback at: {self.csv_path}")
                if os.path.exists(self.csv_path):
                    df = pd.read_csv(self.csv_path)
                    
                    # Apply filters if provided
                    if df is not None and not df.empty:
                        # Find and convert date column
                        date_col = None
                        for col in df.columns:
                            if col.lower() == 'date':
                                date_col = col
                                df[date_col] = pd.to_datetime(df[date_col])
                                break
                        
                        # Filter by years/quarters if requested
                        if date_col and years and len(years) > 0:
                            df = df[df[date_col].dt.year.isin(years)]
                        if date_col and quarters and len(quarters) > 0:
                            df = df[df[date_col].dt.quarter.isin(quarters)]
                    
                    # Generate analysis from CSV data
                    analysis = self._generate_analysis(df, query_text, years, quarters)
                    chart_path = self._generate_chart(df, "close")
                    
                    return {
                        "response": f"Note: Using CSV data as fallback due to database error.\n\n{analysis}",
                        "chart": chart_path,
                        "sources": ["NVIDIA Stock History CSV (Fallback)"]
                    }
                
            except Exception as csv_error:
                logger.error(f"CSV fallback also failed: {str(csv_error)}")
            
            return {
                "response": "I encountered an error while accessing the stock database. The specific error details have been logged for troubleshooting.",
                "chart": None,
                "sources": []
            }
    
    def _generate_sql_with_llm(self, query_text: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> str:
        """Use LLM to generate appropriate SQL query based on the user query"""
        # Prepare context about the database schema
        schema_context = """
        Database: NVIDIA_DB
        Schema: STOCK_DATA
        Table: NVDA_STOCK_DATA
        
        Columns:
        - date (Date): The trading date
        - close (Float): Closing stock price
        - high (Float): Highest stock price during the trading day
        - low (Float): Lowest stock price during the trading day
        - open (Float): Opening stock price
        - volume (Float): Trading volume
        - ma_50 (Float): 50-day moving average
        - ma_200 (Float): 200-day moving average
        - daily_return (Float): Daily return percentage
        - monthly_return (Float): Monthly return percentage
        - yearly_return (Float): Yearly return percentage
        - volatility_21d (Float): 21-day volatility
        
        Note: This table contains NVIDIA's historical stock data, NOT revenue or earnings data.
        """
        
        # Add filter information
        filter_info = ""
        if years and len(years) > 0:
            filter_info += f"\nFilter by years: {years}"
        if quarters and len(quarters) > 0:
            filter_info += f"\nFilter by quarters: {quarters}"
        
        # Create a prompt for SQL generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert in SQL who specializes in financial data analysis. 
            Your task is to translate a natural language query about NVIDIA's stock performance into a valid Snowflake SQL query.
            
            GUIDELINES:
            1. Create a SQL query that extracts the most relevant data to answer the question.
            2. Always include the date column for time series analysis.
            3. Use proper Snowflake SQL syntax.
            4. Apply appropriate functions like EXTRACT() for date parts.
            5. For time period filtering, use clauses like "WHERE EXTRACT(YEAR FROM date) IN (2020, 2021)"
            6. Add appropriate ORDER BY clauses.
            7. If the query requires calculations like growth rates, use window functions.
            8. If analyzing time periods, consider using aggregations with GROUP BY.
            9. Use CTE (WITH clause) for complex queries requiring multiple steps.
            10. For growth analysis, calculate percentage changes between periods.
            11. Return ONLY the SQL query, no explanations or comments.
            
            REMEMBER: The database only contains stock prices and trading information - NOT actual revenue, earnings, or other financial statement data.
            """),
            ("human", f"""
            Database Schema Information:
            {schema_context}
            
            {filter_info}
            
            User's query: {query_text}
            
            Generate a Snowflake SQL query to answer this question. Return ONLY the SQL query itself.
            """)
        ])
        
        # Generate SQL query
        chain = prompt | self.llm
        response = chain.invoke({"query": query_text})
        
        # Extract SQL query from response
        sql_query = response.content.strip()
        
        # Add explicit time filters if not already included in the generated SQL
        if "WHERE" not in sql_query.upper():
            where_clauses = []
            
            if years and len(years) > 0:
                years_str = ", ".join([str(year) for year in years])
                where_clauses.append(f"EXTRACT(YEAR FROM date) IN ({years_str})")
            
            if quarters and len(quarters) > 0:
                quarters_str = ", ".join([str(quarter) for quarter in quarters])
                where_clauses.append(f"EXTRACT(QUARTER FROM date) IN ({quarters_str})")
            
            if where_clauses:
                sql_query += " WHERE " + " AND ".join(where_clauses)
            
            sql_query += " ORDER BY date"
        
        return sql_query
    
    def _generate_fallback_query(self, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> str:
        """Generate a simple fallback query when LLM-generated query fails"""
        # Define the table
        table_name = "NVDA_STOCK_DATA"
        
        # Base SELECT statement with all columns
        base_query = f"""
        SELECT 
            date, 
            close, 
            high, 
            low, 
            open, 
            volume, 
            ma_50, 
            ma_200, 
            daily_return, 
            monthly_return, 
            yearly_return, 
            volatility_21d
        FROM 
            {table_name}
        """
        
        # Add filters for years and quarters if provided
        where_clauses = []
        
        if years and len(years) > 0:
            years_str = ", ".join([str(year) for year in years])
            where_clauses.append(f"EXTRACT(YEAR FROM date) IN ({years_str})")
        
        if quarters and len(quarters) > 0:
            quarters_str = ", ".join([str(quarter) for quarter in quarters])
            where_clauses.append(f"EXTRACT(QUARTER FROM date) IN ({quarters_str})")
        
        # Combine WHERE clauses if any exist
        if where_clauses:
            where_statement = " WHERE " + " AND ".join(where_clauses)
            full_query = base_query + where_statement
        else:
            full_query = base_query
            full_query += " WHERE date >= DATEADD(year, -1, CURRENT_DATE())"
        
        # Add ORDER BY date
        full_query += " ORDER BY date"
        
        return full_query
    
    def _determine_best_chart_metric(self, df: pd.DataFrame, query_text: str) -> str:
        """Determine the best metric to chart based on query and available data"""
        query_lower = query_text.lower()
        
        # Check if a specific metric is mentioned in the query
        if "open" in query_lower:
            return "open"
        elif "high" in query_lower:
            return "high"
        elif "low" in query_lower:
            return "low"
        elif "volume" in query_lower:
            return "volume"
        elif "volatility" in query_lower:
            return "volatility_21d"
        elif "moving average" in query_lower or "ma" in query_lower:
            if "50" in query_lower:
                return "ma_50"
            elif "200" in query_lower:
                return "ma_200"
        
        # Check available columns for specialized metrics
        if "price_change_pct" in df.columns:
            return "price_change_pct"
        elif "period_end" in df.columns:
            return "period_end"
        
        # Default to close price
        return "close"
    
    def _execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Execute the SQL query against Snowflake and return results as a DataFrame
        """
        conn = self.connect_to_snowflake()
        try:
            # Use pandas to execute query and return DataFrame
            df = pd.read_sql(sql_query, conn)
            
            # If no data returned, try a fallback query
            if df.empty:
                logger.info("No data returned from initial query, trying fallback")
                # Try a simpler query without filters if the first one returned no results
                fallback_query = """
                SELECT 
                    date, 
                    close, 
                    high, 
                    low, 
                    open, 
                    volume, 
                    ma_50, 
                    ma_200, 
                    daily_return, 
                    monthly_return, 
                    yearly_return, 
                    volatility_21d
                FROM 
                    NVDA_STOCK_DATA
                ORDER BY date DESC
                LIMIT 30
                """
                df = pd.read_sql(fallback_query, conn)
                logger.info(f"Fallback query returned {len(df)} rows")
                
            return df
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            # Try to read from CSV as final fallback
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NVDA_5yr_history_20250407.csv")
            if os.path.exists(csv_path):
                logger.info(f"Falling back to CSV file at {csv_path}")
                df = pd.read_csv(csv_path)
                # Convert date column to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                return df
            else:
                raise
        finally:
            conn.close()

    def _generate_chart(self, df, metric="close") -> str:
        """Generate chart for visualization"""
        # Find a date column (case insensitive)
        date_col = None
        for col in df.columns:
            if col.lower() == 'date' or col.lower() == 'period':
                date_col = col
                break
                
        if not date_col:
            logger.error("No date or period column found for chart generation.")
            return None
            
        # Find the metric column (case insensitive)
        metric_col = None
        for col in df.columns:
            if col.lower() == metric.lower():
                metric_col = col
                break
        
        if not metric_col:
            # Look for alternative metric columns
            for col in df.columns:
                if col.lower() in ['close', 'closing', 'price', 'value', 'period_end', 'end_price', 'average_price']:
                    metric_col = col
                    logger.info(f"Using alternative metric column: {metric_col}")
                    break
        
        if not metric_col:
            logger.error(f"Could not find metric column for {metric}. Available columns: {df.columns.tolist()}")
            return None
        
        # Sort by date
        if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.sort_values(date_col)
        
        # Build the figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Use different chart types based on data
        if 'period' in df.columns and len(df) <= 12:  # For quarterly/yearly summary data
            # Bar chart for period data
            periods = df['period'].tolist()
            values = df[metric_col].tolist()
            
            bars = ax.bar(periods, values, color='#76b900', alpha=0.8)  # NVIDIA green
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'${height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
                
            chart_title = "NVIDIA Stock Performance by Period"
        else:  # For daily data or other time series
            # Line chart for time series
            ax.plot(df[date_col], df[metric_col], marker="o", linewidth=2, color="#76b900")  # NVIDIA green
            
            # Add data labels for some points (not all to avoid overcrowding)
            n = max(1, len(df) // 10)  # Show about 10 labels
            for i, row in df.iloc[::n].iterrows():
                try:
                    ax.annotate(
                        f'${row[metric_col]:.2f}',
                        (row[date_col], row[metric_col]),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha='center',
                        fontsize=8,
                        color='gray'
                    )
                except:
                    pass  # Skip annotation if it fails
                
            chart_title = f"NVIDIA {metric_col.upper()} Stock Price"

        # Title and axes
        ax.set_title(chart_title, fontsize=14)
        ax.set_xlabel("Date/Period", fontsize=12)
        ax.set_ylabel(f"Price ($)", fontsize=12)
        plt.xticks(rotation=45)
        ax.grid(True, linestyle="--", alpha=0.6)

        # Format y-axis with dollar formatting
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.2f}"))

        # Save the chart to the charts directory
        chart_filename = f"nvda_{metric_col.lower()}_chart.png"
        chart_path = os.path.join(self.charts_dir, chart_filename)
        
        plt.tight_layout()
        try:
            plt.savefig(chart_path)
            logger.info(f"Chart saved to: {chart_path}")
            plt.close()
            
            # Return the absolute path for the frontend to find
            return os.path.abspath(chart_path)
        except Exception as e:
            logger.error(f"Error saving chart: {str(e)}")
            plt.close()
            return None
        
    def _generate_analysis(self, df: pd.DataFrame, query_text: str, 
                          years: Optional[List[int]] = None, quarters: Optional[List[int]] = None,
                          sql_query: str = None) -> str:
        """Generate analysis of financial data using LLM"""
        # Format dataframe for context
        if len(df) <= 20:  # If small dataset, include all rows
            df_str = df.to_string()
        else:  # For larger datasets, include a selection of rows
            # Get first few rows
            first_rows = df.head(5).to_string()
            # Get middle rows
            mid_index = len(df) // 2
            middle_rows = df.iloc[mid_index-2:mid_index+3].to_string()
            # Get last few rows
            last_rows = df.tail(5).to_string()
            
            df_str = f"First rows:\n{first_rows}\n\nMiddle rows:\n{middle_rows}\n\nLast rows:\n{last_rows}"
        
        # Create a filter description for context
        filter_desc = ""
        if years and len(years) > 0:
            filter_desc += f"Years: {', '.join(map(str, years))}\n"
        if quarters and len(quarters) > 0:
            filter_desc += f"Quarters: {', '.join([f'Q{q}' for q in quarters])}\n"
        
        # Get column info for context
        column_info = f"Available columns: {', '.join(df.columns.tolist())}\n"
        
        # Basic stats for better context
        stats_info = "Data statistics:\n"
        
        # Row count
        stats_info += f"- Total rows: {len(df)}\n"
        
        # Date range if available
        if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
            stats_info += f"- Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\n"
        
        # Perform deeper statistical analysis relevant to the query
        query_lower = query_text.lower()
        statistical_analysis = "Statistical insights:\n"
        
        # Price analysis
        if 'close' in df.columns:
            if len(df) > 1:
                start_price = df.iloc[0]['close']
                end_price = df.iloc[-1]['close']
                change_pct = ((end_price - start_price) / start_price) * 100
                statistical_analysis += f"- Overall price change: {change_pct:.2f}% (from ${start_price:.2f} to ${end_price:.2f})\n"
            
            if len(df) > 10:  # Only calculate volatility for sufficient data
                volatility = df['close'].pct_change().std() * (252**0.5) * 100  # Annualized volatility
                statistical_analysis += f"- Annualized volatility: {volatility:.2f}%\n"
        
        # Volume analysis
        if 'volume' in df.columns:
            avg_volume = df['volume'].mean()
            max_volume_idx = df['volume'].idxmax()
            max_volume_date = df.loc[max_volume_idx, 'date'] if 'date' in df.columns else 'Unknown'
            statistical_analysis += f"- Average trading volume: {avg_volume:.2f}\n"
            statistical_analysis += f"- Highest volume: {df['volume'].max():.2f} on {max_volume_date}\n"
        
        # Include SQL query if provided
        query_info = f"SQL Query used:\n{sql_query}\n\n" if sql_query else ""
        
        # Load the standalone Snowflake prompt
        snowflake_prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "prompts", "snowflake_agent_standalone.txt")
        
        # Use default prompt if the file doesn't exist
        if os.path.exists(snowflake_prompt_path):
            with open(snowflake_prompt_path, 'r') as f:
                system_prompt = f.read()
        else:
            system_prompt = """
            You are NVIDIA Financial Analyst, an expert in stock market analysis and financial reporting.
            
            TASK:
            Create an in-depth, professional financial analysis of NVIDIA stock data that answers the user's query with unparalleled detail and insight.
            
            DATA ANALYSIS REQUIREMENTS:
            1. DEPTH OF ANALYSIS: Provide sophisticated, thorough analysis that includes:
               - Detailed technical analysis of price movements and patterns
               - Comprehensive statistical evaluation of key metrics
               - Multi-timeframe comparisons where relevant (daily, weekly, monthly trends)
               - Correlation analysis between different metrics when available
               - Advanced financial ratio calculations where applicable
            
            2. INSIGHT REQUIREMENTS:
               - Identify specific inflection points and their potential causes
               - Analyze momentum and trend strength with appropriate indicators
               - Provide context for abnormal trading periods or outliers
               - Discuss the implications of the data patterns for NVIDIA's business
               - Connect stock performance to relevant company or industry events
            
            3. VISUALIZATION GUIDANCE (for charts already created):
               - Interpret key support/resistance levels visible in charts
               - Identify important pattern formations (head and shoulders, channels, etc.)
               - Explain the significance of moving average crossovers or divergences
               - Note volume confirmation or divergence from price action
            
            4. COMPARATIVE CONTEXT:
               - Benchmark against relevant index performance where data allows
               - Compare current periods to historical precedents
               - Evaluate performance relative to semiconductor industry standards
               - Identify outperformance or underperformance periods
            
            FORMAT AND STRUCTURE:
            1. DETAILED EXECUTIVE SUMMARY: Concise but comprehensive overview (2-3 paragraphs)
            
            2. PRICE ACTION ANALYSIS: In-depth breakdown of price movements
               - Trend analysis (primary, secondary, and tertiary trends)
               - Support/resistance identification
               - Pattern recognition and implications
               - Price momentum analysis
            
            3. VOLUME PROFILE ASSESSMENT: Detailed volume analysis
               - Volume trends and anomalies
               - Volume-price relationship
               - Accumulation/distribution patterns
               - Institutional activity indicators
            
            4. VOLATILITY EXAMINATION: Comprehensive volatility insights
               - Historical volatility trends
               - Volatility regime identification
               - Risk assessment based on volatility metrics
               - Volatility comparison to relevant benchmarks
            
            5. TECHNICAL INDICATOR DEEP DIVE: Analysis of available technical indicators
               - Moving average relationships and crossovers
               - Relative strength evaluation
               - Momentum indicator analysis
               - Trend confirmation or divergence signals
            
            6. FUNDAMENTAL CORRELATION: Connect price action to business performance
               - Stock behavior around key company announcements
               - Price reaction to industry developments
               - Sentiment analysis based on price-volume behavior
               - Potential future catalysts suggested by technical patterns
            
            7. CONCLUSION WITH ACTIONABLE INSIGHTS: Summary of key findings (1-2 paragraphs)
            
            STYLE GUIDELINES:
            - Use precise, technical financial language
            - Include exact figures with appropriate units and formatting
            - Bold significant metrics and key insights
            - Use professional financial analysis terminology
            - Maintain objective, data-driven tone throughout
            - Use bullet points strategically for important metrics
            
            REMEMBER: This analysis will be used by sophisticated financial professionals, so provide depth and precision that would satisfy expert analysts.
            """
        
        # Create prompt for analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"""
            USER QUERY: "{query_text}"
            
            TIME FILTERS:
            {filter_desc}
            
            DATA OVERVIEW:
            {column_info}
            {stats_info}
            {statistical_analysis}
            
            SQL QUERY AND SAMPLING:
            {query_info}
            
            DATA SAMPLE:
            {df_str}
            
            Please provide an exceptionally detailed and insightful analysis that thoroughly addresses the query.
            Include exact figures, comprehensive technical analysis, and sophisticated interpretation of patterns and trends.
            This is for a professional research report, so depth and precision are essential.
            """)
        ])
        
        # Generate analysis
        chain = prompt | self.llm
        response = chain.invoke({
            "data": df_str,
            "column_info": column_info,
            "filter_desc": filter_desc,
            "stats_info": stats_info,
            "statistical_analysis": statistical_analysis,
            "query": query_text
        })
        
        return response.content