import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any, Optional, List, Tuple, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import snowflake.connector
import base64
from io import BytesIO
import logging
import functools
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnowflakeAgent:
    def __init__(self):
        self._init_environment()
        self._init_connections()
        self._init_paths()
        
        # Cache for storing query results to avoid redundant database calls
        self.cache = {}
        self.cache_ttl = 3600  # Cache time-to-live in seconds (1 hour)
        
    def _init_environment(self):
        """Initialize environment variables with root directory priority"""
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
        
        # Get Snowflake credentials
        self.snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.snowflake_user = os.getenv("SNOWFLAKE_USER")
        self.snowflake_password = os.getenv("SNOWFLAKE_PASSWORD")
        self.snowflake_database = os.getenv("SNOWFLAKE_DATABASE", "NVIDIA_DB")
        self.snowflake_schema = os.getenv("SNOWFLAKE_SCHEMA", "STOCK_DATA")
        self.snowflake_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        self.snowflake_role = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
        
        logger.info(f"Initializing Snowflake connection with account: {self.snowflake_account}, user: {self.snowflake_user}")
        logger.info(f"Database: {self.snowflake_database}, Schema: {self.snowflake_schema}")
    
    def _init_connections(self):
        """Initialize LLM and test Snowflake connection"""
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0)
        
        # Test connection to Snowflake
        try:
            self._test_connection()
            logger.info("Successfully connected to Snowflake")
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}. Will use CSV fallback if needed.")
            
    def _init_paths(self):
        """Initialize file paths for CSV fallback and chart output"""
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
            warehouse=self.snowflake_warehouse,
            role=self.snowflake_role
        )
    
    def query(self, query_text: str, years: Optional[List[int]] = None, 
              quarters: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Query NVIDIA financial data based on query text, years, and quarters.
        Uses SQL to query the Snowflake database with LLM-generated queries.
        """
        start_time = time.time()
        
        try:
            # Check cache for identical query
            cache_key = f"{query_text}_{years}_{quarters}"
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    logger.info(f"Using cached result for query: {query_text}")
                    return cache_entry['result']
            
            # Handle revenue-related queries specially
            if self._is_revenue_query(query_text):
                result = self._handle_revenue_query(query_text, years, quarters)
                if result:
                    self._update_cache(cache_key, result)
                    return result
            
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
                result = {
                    "response": f"No stock data found for NVIDIA with the specified filters (Years: {years}, Quarters: {quarters}). Please try different time periods or a broader query.",
                    "chart": None,
                    "sources": ["Snowflake - NVIDIA_DB.STOCK_DATA.NVDA_STOCK_DATA"],
                    "execution_time": f"{time.time() - start_time:.2f}s"
                }
                self._update_cache(cache_key, result)
                return result
            
            # Generate chart if data is available
            chart_path = None
            if not df.empty:
                chart_metric = self._determine_best_chart_metric(df, query_text)
                chart_path = self._generate_chart(df, chart_metric)
            
            # Generate analysis with LLM
            analysis = self._generate_analysis(df, query_text, years, quarters, sql_query)
            
            result = {
                "response": analysis,
                "chart": chart_path,
                "sources": ["Snowflake - NVIDIA_DB.STOCK_DATA.NVDA_STOCK_DATA"],
                "execution_time": f"{time.time() - start_time:.2f}s"
            }
            
            self._update_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}", exc_info=True)
            
            # Try to use CSV as fallback
            try:
                result = self._try_csv_fallback(query_text, years, quarters, start_time)
                if result:
                    return result
            except Exception as csv_error:
                logger.error(f"CSV fallback also failed: {str(csv_error)}")
            
            return {
                "response": "I encountered an error while accessing the stock database. The specific error details have been logged for troubleshooting.",
                "chart": None,
                "sources": [],
                "execution_time": f"{time.time() - start_time:.2f}s"
            }
    
    def _update_cache(self, key: str, result: Dict[str, Any]):
        """Update the cache with a new result"""
        self.cache[key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def _is_revenue_query(self, query_text: str) -> bool:
        """Check if the query is related to revenue"""
        revenue_terms = ["revenue", "earnings", "income", "profit", "sales"]
        return any(term in query_text.lower() for term in revenue_terms)
    
    def _handle_revenue_query(self, query_text: str, years: Optional[List[int]] = None,
                             quarters: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
        """Special handling for revenue-related queries"""
        revenue_note = (
            "Note: While I don't have direct revenue figures from financial statements in the database, "
            "I've approximated revenue trends using stock metrics. For true revenue figures, "
            "please refer to NVIDIA's official quarterly reports or financial statements. "
            "This approximation is based on stock price and trading volume patterns over time."
        )
        
        # Only proceed with specialized handling if we have years to filter by
        if not years or len(years) < 2:
            return None
            
        # Try to generate an approximation using stock data as a proxy
        simplified_query = f"""
        WITH yearly_data AS (
            SELECT 
                EXTRACT(YEAR FROM date) AS year,
                AVG(close) AS avg_price,
                SUM(volume) AS total_volume,
                AVG(close * volume) AS avg_daily_value
            FROM NVDA_STOCK_DATA
            WHERE EXTRACT(YEAR FROM date) IN ({', '.join(map(str, years))})
            GROUP BY year
            ORDER BY year
        )
        SELECT 
            year,
            avg_daily_value * 252 AS revenue,  -- Approximation using annual trading days
            LAG(avg_daily_value * 252) OVER (ORDER BY year) AS prev_revenue,
            (avg_daily_value * 252 - LAG(avg_daily_value * 252) OVER (ORDER BY year)) / 
                NULLIF(LAG(avg_daily_value * 252) OVER (ORDER BY year), 0) * 100 AS revenue_growth_rate
        FROM yearly_data;
        """
        try:
            approx_data = self._execute_query(simplified_query)
            if approx_data is None or approx_data.empty:
                return None
                
            chart_path = self._generate_chart(approx_data, "revenue")
            
            # Generate enhanced analysis for revenue approximation
            analysis = self._generate_analysis(
                approx_data, 
                f"Approximate NVIDIA revenue trends for years {', '.join(map(str, years))}", 
                years, 
                quarters, 
                simplified_query
            )
            
            return {
                "response": f"{analysis}\n\n{revenue_note}",
                "chart": chart_path,
                "sources": ["Snowflake - NVIDIA_DB.STOCK_DATA.NVDA_STOCK_DATA (Approximated Revenue)"]
            }
        except Exception as e:
            logger.error(f"Error generating revenue approximation: {str(e)}")
            return None
    
    def _generate_sql_with_llm(self, query_text: str, years: Optional[List[int]] = None, 
                              quarters: Optional[List[int]] = None) -> str:
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
        
        # Check if the query is about revenue trends specifically
        is_revenue_query = "revenue" in query_text.lower() and ("trend" in query_text.lower() or "growth" in query_text.lower())
        
        # Select appropriate prompt based on query type
        if is_revenue_query and years and len(years) >= 2:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are an expert in SQL who specializes in financial data analysis.
                Your task is to create a query that approximates NVIDIA's revenue trends using stock market data.
                
                GUIDELINES:
                1. Create a query that APPROXIMATES revenue growth from stock price and volume data.
                2. Include yearly aggregation with year extracted from the date column.
                3. Calculate a proxy for revenue using price * volume as a base metric.
                4. Calculate year-over-year growth rates using LAG() window functions.
                5. Return revenue approximation and growth rate percentages by year.
                6. Use proper Snowflake SQL syntax with a WITH clause for intermediate calculations.
                7. Return ONLY the SQL query, no explanations or comments.
                
                IMPORTANT: We're approximating revenue since the database only contains stock market data, not actual financial statement data.
                """),
                ("human", f"""
                Database Schema Information:
                {schema_context}
                
                {filter_info}
                
                Generate a Snowflake SQL query to approximate NVIDIA's revenue trends by year using stock market data.
                Return ONLY the SQL query itself.
                """)
            ])
        else:        
            # Default prompt for other queries
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
        return self._add_time_filters_if_needed(sql_query, years, quarters)
    
    def _add_time_filters_if_needed(self, sql_query: str, years: Optional[List[int]], 
                                   quarters: Optional[List[int]]) -> str:
        """Add time filters to the SQL query if not already present"""
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
    
    def _generate_fallback_query(self, years: Optional[List[int]] = None, 
                                quarters: Optional[List[int]] = None) -> str:
        """Generate a simple fallback query when LLM-generated query fails"""
        # Base SELECT statement with all columns
        base_query = """
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
        """
        
        # Add filters and ORDER BY
        return self._add_time_filters_if_needed(base_query, years, quarters)
    
    def _determine_best_chart_metric(self, df: pd.DataFrame, query_text: str) -> str:
        """Determine the best metric to chart based on query and available data"""
        query_lower = query_text.lower()
        
        # Priority 1: Check explicit mentions in the query
        metric_mappings = {
            "open": ["open"],
            "high": ["high", "highest"],
            "low": ["low", "lowest"],
            "close": ["close", "closing"],
            "volume": ["volume", "trading volume"],
            "volatility_21d": ["volatility", "volatile"],
            "ma_50": ["50", "50-day", "50 day", "moving average", "ma"],
            "ma_200": ["200", "200-day", "200 day", "moving average", "ma"],
            "revenue": ["revenue", "sales", "earnings"],
            "revenue_growth_rate": ["growth rate", "revenue growth", "growth percentage"]
        }
        
        # Check for each metric in the query text
        for metric, keywords in metric_mappings.items():
            if any(keyword in query_lower for keyword in keywords):
                # Also verify column exists (or close equivalent)
                if metric in df.columns:
                    return metric
                # Check for similar column names
                for col in df.columns:
                    if metric.lower() in col.lower():
                        return col
        
        # Priority 2: Check for specialized metrics in available columns
        specialized_metrics = [
            "revenue", "revenue_growth_rate", "price_change_pct", 
            "period_end", "growth", "return"
        ]
        
        for metric in specialized_metrics:
            for col in df.columns:
                if metric.lower() in col.lower():
                    return col
        
        # Priority 3: Default to close price if available
        if "close" in df.columns:
            return "close"
        
        # Final fallback: Use the first numeric column
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
                
        # Absolute last resort
        return df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    def _execute_query(self, sql_query: str) -> Optional[pd.DataFrame]:
        """Execute the SQL query against Snowflake and return results as a DataFrame"""
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
            return self._read_from_csv()
        finally:
            conn.close()
            
    def _read_from_csv(self) -> Optional[pd.DataFrame]:
        """Read NVIDIA data from CSV as fallback"""
        if os.path.exists(self.csv_path):
            logger.info(f"Falling back to CSV file at {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            logger.error(f"CSV fallback file not found at {self.csv_path}")
            return None
            
    def _try_csv_fallback(self, query_text: str, years: Optional[List[int]], 
                         quarters: Optional[List[int]], start_time: float) -> Optional[Dict[str, Any]]:
        """Try to use CSV data as fallback for the query"""
        logger.info(f"Attempting to use CSV fallback at: {self.csv_path}")
        if not os.path.exists(self.csv_path):
            return None
            
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
        chart_metric = self._determine_best_chart_metric(df, query_text)
        chart_path = self._generate_chart(df, chart_metric)
        
        return {
            "response": f"Note: Using CSV data as fallback due to database error.\n\n{analysis}",
            "chart": chart_path,
            "sources": ["NVIDIA Stock History CSV (Fallback)"],
            "execution_time": f"{time.time() - start_time:.2f}s"
        }

    def _generate_chart(self, df: pd.DataFrame, metric: str = "close") -> Optional[str]:
        """Generate chart for visualization with optimized handling of different data formats"""
        try:
            # Check for aggregated data (by year) and convert year to a date format
            if 'year' in df.columns and not any(col.lower() == 'date' or col.lower() == 'period' for col in df.columns):
                logger.info("Aggregated data detected. Converting 'year' to date format.")
                # Convert year to a date format
                df['date'] = pd.to_datetime(df['year'], format='%Y')
                date_col = 'date'
    
                # Determine the appropriate metric column
                metric_col = self._get_metric_column(df, metric)
                if not metric_col:
                    return None
            else:
                # Use standard approach for non-aggregated data
                # Find a date column (case insensitive)
                date_col = self._find_column(df, ['date', 'period'])
                if not date_col:
                    logger.error("No date or period column found for chart generation.")
                    return None
                    
                # Find the metric column
                metric_col = self._get_metric_column(df, metric)
                if not metric_col:
                    return None
            
            # Sort by date
            if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df = df.sort_values(date_col)
            
            # Build the figure
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Use different chart types based on data
            is_period_data = ('period' in df.columns and len(df) <= 12) or ('year' in df.columns and len(df) <= 10)
            
            if is_period_data:  # For quarterly/yearly summary data
                self._create_bar_chart(df, ax, date_col, metric_col)
            else:  # For daily data or other time series
                self._create_line_chart(df, ax, date_col, metric_col)
                
            # Format axis labels and styling based on metric type
            self._format_chart_axes(ax, metric_col)
            
            # Save the chart
            return self._save_chart(fig, metric_col)
                
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}", exc_info=True)
            plt.close()
            return None
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find a column in the dataframe that matches any of the possible names"""
        for col in df.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                return col
        return None
    
    def _get_metric_column(self, df: pd.DataFrame, metric: str) -> Optional[str]:
        """Find the appropriate metric column in the dataframe"""
        # Direct match for the requested metric
        for col in df.columns:
            if col.lower() == metric.lower():
                return col
        
        # For revenue queries, check for revenue-related columns
        if metric.lower() == 'revenue':
            revenue_columns = ['revenue', 'sales', 'income']
            for col in df.columns:
                if any(rc in col.lower() for rc in revenue_columns):
                    logger.info(f"Using '{col}' as revenue metric")
                    return col
        
        # For growth rate queries, check for growth-related columns
        if 'growth' in metric.lower():
            growth_columns = ['growth', 'growth_rate', 'change']
            for col in df.columns:
                if any(gc in col.lower() for gc in growth_columns):
                    logger.info(f"Using '{col}' as growth metric")
                    return col
        
        # Look for alternative price-related columns
        price_alternatives = ['close', 'closing', 'price', 'value', 'period_end', 'end_price', 'average_price']
        for col in df.columns:
            if col.lower() in price_alternatives:
                logger.info(f"Using alternative metric column: {col}")
                return col
        
        # Last resort - use the first numeric column that's not a date/year
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col.lower() not in ['date', 'year', 'period', 'quarter']:
                logger.info(f"Using fallback numeric column: {col}")
                return col
                
        logger.error(f"Could not find appropriate metric column for {metric}. Available columns: {df.columns.tolist()}")
        return None
    
    def _create_bar_chart(self, df: pd.DataFrame, ax, date_col: str, metric_col: str):
        """Create a bar chart for period data"""
        # Determine x-axis labels
        if 'period' in df.columns:
            labels = df['period'].tolist()
            chart_title = "NVIDIA Stock Performance by Period"
        else:
            # Use year column or format date column to year
            if 'year' in df.columns:
                labels = [str(year) for year in df['year'].tolist()]
            else:
                labels = [date.strftime('%Y') for date in df[date_col].tolist()]
            
            # Set chart title based on the metric
            chart_title = self._get_chart_title(metric_col)
        
        values = df[metric_col].tolist()
        
        # Create the bar chart
        bars = ax.bar(labels, values, color='#76b900', alpha=0.8)  # NVIDIA green
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            # Format the label based on metric type
            label = self._format_value_label(height, metric_col)
                
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
        ax.set_title(chart_title, fontsize=14)
    
    def _create_line_chart(self, df: pd.DataFrame, ax, date_col: str, metric_col: str):
        """Create a line chart for time series data"""
        # Line chart for time series
        ax.plot(df[date_col], df[metric_col], marker="o", linewidth=2, color="#76b900")  # NVIDIA green
        
        # Determine chart title
        chart_title = self._get_chart_title(metric_col)
        
        # Add data labels for some points (not all to avoid overcrowding)
        n = max(1, len(df) // 10)  # Show about 10 labels
        for i, row in df.iloc[::n].iterrows():
            try:
                # Format the label
                label = self._format_value_label(row[metric_col], metric_col)
                    
                ax.annotate(
                    label,
                    (row[date_col], row[metric_col]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha='center',
                    fontsize=8,
                    color='gray'
                )
            except:
                pass  # Skip annotation if it fails
                
        ax.set_title(chart_title, fontsize=14)
    
    def _get_chart_title(self, metric_col: str) -> str:
        """Generate appropriate chart title based on metric column"""
        if metric_col.lower() == 'close':
            return "NVIDIA Stock Closing Price"
        elif metric_col.lower() == 'open':
            return "NVIDIA Stock Opening Price"
        elif metric_col.lower() == 'high':
            return "NVIDIA Stock High Price"
        elif metric_col.lower() == 'low':
            return "NVIDIA Stock Low Price"
        elif metric_col.lower() == 'volume':
            return "NVIDIA Trading Volume"
        elif metric_col.lower() == 'ma_50':
            return "NVIDIA 50-Day Moving Average"
        elif metric_col.lower() == 'ma_200':
            return "NVIDIA 200-Day Moving Average"
        elif 'revenue' in metric_col.lower():
            return "NVIDIA Annual Revenue"
        elif 'growth' in metric_col.lower():
            return "NVIDIA Annual Growth Rate (%)"
        elif 'return' in metric_col.lower():
            return f"NVIDIA {metric_col.replace('_', ' ').title()}"
        else:
            return f"NVIDIA {metric_col.replace('_', ' ').title()}"
    
    def _format_value_label(self, value: float, metric_col: str) -> str:
        """Format value label based on metric type"""
        if 'growth' in metric_col.lower() or 'return' in metric_col.lower():
            return f'{value:.2f}%'
        elif 'revenue' in metric_col.lower():
            if value > 1_000_000_000:  # If in billions
                return f'${value/1_000_000_000:.2f}B'
            elif value > 1_000_000:  # If in millions
                return f'${value/1_000_000:.2f}M'
            else:
                return f'${value:.2f}'
        elif 'volume' in metric_col.lower():
            if value > 1_000_000:  # If in millions
                return f'{value/1_000_000:.2f}M'
            elif value > 1_000:  # If in thousands
                return f'{value/1_000:.2f}K'
            else:
                return f'{value:.0f}'
        else:
            return f'${value:.2f}'
    
    def _format_chart_axes(self, ax, metric_col: str):
        """Format chart axes based on metric type"""
        # Set appropriate x-axis label
        ax.set_xlabel("Date/Period", fontsize=12)
        
        # Set appropriate y-axis label and formatter
        if metric_col.lower() == 'revenue_growth_rate' or 'growth' in metric_col.lower() or 'return' in metric_col.lower():
            ax.set_ylabel(f"Percentage (%)", fontsize=12)
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.2f}%"))
        elif metric_col.lower() == 'revenue':
            ax.set_ylabel(f"Revenue ($)", fontsize=12)
            
            # Custom formatter for billions/millions
            def revenue_formatter(x, pos):
                if x >= 1_000_000_000:
                    return f"${x/1_000_000_000:.1f}B"
                elif x >= 1_000_000:
                    return f"${x/1_000_000:.1f}M"
                else:
                    return f"${x:,.0f}"
                    
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(revenue_formatter))
        elif metric_col.lower() == 'volume':
            ax.set_ylabel(f"Volume", fontsize=12)
            
            # Custom formatter for volume
            def volume_formatter(x, pos):
                if x >= 1_000_000_000:
                    return f"{x/1_000_000_000:.1f}B"
                elif x >= 1_000_000:
                    return f"{x/1_000_000:.1f}M"
                elif x >= 1_000:
                    return f"{x/1_000:.1f}K"
                else:
                    return f"{x:,.0f}"
                    
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(volume_formatter))
        else:
            ax.set_ylabel(f"Price ($)", fontsize=12)
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.2f}"))
            
        plt.xticks(rotation=45)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
    
    def _save_chart(self, fig, metric_col: str) -> Optional[str]:
        """Save chart to file and return path"""
        chart_filename = f"nvda_{metric_col.lower()}_chart_{int(time.time())}.png"
        chart_path = os.path.join(self.charts_dir, chart_filename)
        
        try:
            plt.savefig(chart_path)
            logger.info(f"Chart saved to: {chart_path}")
            plt.close(fig)
            
            # Return the absolute path for the frontend to find
            return os.path.abspath(chart_path)
        except Exception as e:
            logger.error(f"Error saving chart: {str(e)}")
            plt.close(fig)
            return None

    def _generate_analysis(self, df: pd.DataFrame, query_text: str, 
                          years: Optional[List[int]] = None, quarters: Optional[List[int]] = None,
                          sql_query: str = None) -> str:
        """Generate analysis of financial data using LLM with optimized context preparation"""
        # Format dataframe efficiently for context
        df_context = self._prepare_dataframe_context(df)
        
        # Create a filter description for context
        filter_desc = self._prepare_filter_description(years, quarters)
        
        # Get column info and statistics
        column_info, stats_info = self._prepare_data_statistics(df)
        
        # Perform deeper statistical analysis relevant to the query
        statistical_analysis = self._prepare_statistical_analysis(df, query_text)
        
        # Include SQL query if provided
        query_info = f"SQL Query used:\n{sql_query}\n\n" if sql_query else ""
        
        # Load or use default prompt template
        prompt_template = self._load_prompt_template()
        
        # Create prompt for analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
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
            {df_context}
            
            Please provide an exceptionally detailed and insightful analysis that thoroughly addresses the query.
            Include exact figures, comprehensive technical analysis, and sophisticated interpretation of patterns and trends.
            This is for a professional research report, so depth and precision are essential.
            """)
        ])
        
        # Generate analysis
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "data": df_context,
                "column_info": column_info,
                "filter_desc": filter_desc,
                "stats_info": stats_info,
                "statistical_analysis": statistical_analysis,
                "query": query_text
            })
            
            return response.content
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return f"Error generating analysis: {str(e)}"
    
    def _prepare_dataframe_context(self, df: pd.DataFrame) -> str:
        """Prepare dataframe context in an efficient way"""
        if len(df) <= 20:  # If small dataset, include all rows
            return df.to_string()
        
        # For larger datasets, include a strategic selection
        first_rows = df.head(5).to_string()
        mid_index = len(df) // 2
        middle_rows = df.iloc[mid_index-2:mid_index+3].to_string()
        last_rows = df.tail(5).to_string()
        
        return f"First rows:\n{first_rows}\n\nMiddle rows:\n{middle_rows}\n\nLast rows:\n{last_rows}"
    
    def _prepare_filter_description(self, years: Optional[List[int]], quarters: Optional[List[int]]) -> str:
        """Create a filter description string"""
        filter_desc = ""
        if years and len(years) > 0:
            filter_desc += f"Years: {', '.join(map(str, years))}\n"
        if quarters and len(quarters) > 0:
            filter_desc += f"Quarters: {', '.join([f'Q{q}' for q in quarters])}\n"
        return filter_desc
    
    def _prepare_data_statistics(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Prepare column info and basic statistics"""
        # Column information
        column_info = f"Available columns: {', '.join(df.columns.tolist())}\n"
        
        # Basic stats
        stats_info = "Data statistics:\n"
        stats_info += f"- Total rows: {len(df)}\n"
        
        # Date range if available
        if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
            stats_info += f"- Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\n"
        elif 'year' in df.columns:
            stats_info += f"- Year range: {df['year'].min()} to {df['year'].max()}\n"
            
        return column_info, stats_info
    
    def _prepare_statistical_analysis(self, df: pd.DataFrame, query_text: str) -> str:
        """Prepare statistical analysis based on the query and available data"""
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
        
        # Revenue analysis
        if 'revenue' in df.columns and len(df) > 1:
            start_rev = df.iloc[0]['revenue']
            end_rev = df.iloc[-1]['revenue']
            change_pct = ((end_rev - start_rev) / start_rev) * 100
            
            if start_rev > 1_000_000_000:  # If in billions
                statistical_analysis += f"- Revenue growth: {change_pct:.2f}% (from ${start_rev/1_000_000_000:.2f}B to ${end_rev/1_000_000_000:.2f}B)\n"
            elif start_rev > 1_000_000:  # If in millions
                statistical_analysis += f"- Revenue growth: {change_pct:.2f}% (from ${start_rev/1_000_000:.2f}M to ${end_rev/1_000_000:.2f}M)\n"
            else:
                statistical_analysis += f"- Revenue growth: {change_pct:.2f}% (from ${start_rev:.2f} to ${end_rev:.2f})\n"
        
        # Growth rate analysis
        if 'revenue_growth_rate' in df.columns:
            avg_growth = df['revenue_growth_rate'].mean()
            max_growth = df['revenue_growth_rate'].max()
            min_growth = df['revenue_growth_rate'].min()
            statistical_analysis += f"- Average revenue growth rate: {avg_growth:.2f}%\n"
            statistical_analysis += f"- Highest growth rate: {max_growth:.2f}%\n"
            statistical_analysis += f"- Lowest growth rate: {min_growth:.2f}%\n"
        
        # Volume analysis
        if 'volume' in df.columns:
            avg_volume = df['volume'].mean()
            max_volume_idx = df['volume'].idxmax()
            
            # Get date of max volume if available
            max_volume_date = "Unknown"
            if 'date' in df.columns:
                max_volume_date = df.loc[max_volume_idx, 'date']
                if isinstance(max_volume_date, pd.Timestamp):
                    max_volume_date = max_volume_date.strftime('%Y-%m-%d')
            elif 'year' in df.columns:
                max_volume_date = f"Year {df.loc[max_volume_idx, 'year']}"
                
            if avg_volume > 1_000_000:
                statistical_analysis += f"- Average trading volume: {avg_volume/1_000_000:.2f}M\n"
                statistical_analysis += f"- Highest volume: {df['volume'].max()/1_000_000:.2f}M on {max_volume_date}\n"
            else:
                statistical_analysis += f"- Average trading volume: {avg_volume:.2f}\n"
                statistical_analysis += f"- Highest volume: {df['volume'].max():.2f} on {max_volume_date}\n"
        
        return statistical_analysis
    
    def _load_prompt_template(self) -> str:
        """Load the analysis prompt template from file or use default"""
        snowflake_prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "prompts", "snowflake_agent_standalone.txt")
        
        # Use default prompt if the file doesn't exist
        if os.path.exists(snowflake_prompt_path):
            with open(snowflake_prompt_path, 'r') as f:
                return f.read()
        else:
            return """
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