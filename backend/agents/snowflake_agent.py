# backend/agents/snowflake_agent.py
import os
import pandas as pd
import snowflake.connector
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import base64
from io import BytesIO

class SnowflakeAgent:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Validate required environment variables
        required_vars = [
            "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                raise ValueError(f"{var} environment variable not set")
        
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0)
        
        # Use the correct table name based on screenshot
        self.table_name = os.getenv("SNOWFLAKE_TABLE", "NVDA_STOCK_DATA")
        
        # Create a shared directory for charts
        self.charts_dir = os.getenv("CHARTS_DIR", "/app/shared")
        os.makedirs(self.charts_dir, exist_ok=True)
    
    def query(self, query_text: str, years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Query Snowflake for NVIDIA financial data based on query text, years, and quarters.
        
        Args:
            query_text: The query text
            years: Optional list of years to filter by
            quarters: Optional list of quarters to filter by
            
        Returns:
            Dictionary with structured data, charts, and generated response
        """
        try:
            # Build SQL query with filters
            sql_query = f"SELECT * FROM {self.table_name}"
            
            where_clauses = []
            if years is not None and len(years) > 0:
                # For stock data, we need to extract the year from the DATE column
                years_str = ", ".join([str(year) for year in years])
                where_clauses.append(f"YEAR(DATE) IN ({years_str})")
            
            if quarters is not None and len(quarters) > 0:
                # Extract quarter from DATE
                quarters_str = ", ".join([str(quarter) for quarter in quarters])
                where_clauses.append(f"QUARTER(DATE) IN ({quarters_str})")
                
            if where_clauses:
                sql_query += " WHERE " + " AND ".join(where_clauses)
            
            # Query Snowflake
            df = self._query_snowflake(sql_query)
            
            if df.empty:
                return {
                    "response": f"No stock data found for NVIDIA with the specified filters.",
                    "chart": None,
                    "sources": ["Snowflake Database"]
                }
            
            # Generate chart if data is available
            chart_path = None
            if not df.empty:
                chart_path = self._generate_chart(df, "CLOSE")
            
            # Generate analysis with LLM
            analysis = self._generate_analysis(df, query_text, years, quarters)
            
            return {
                "response": analysis,
                "chart": chart_path,
                "sources": ["Snowflake Database - " + self.table_name]
            }
            
        except Exception as e:
            # Return error details with traceback for better debugging
            import traceback
            error_message = f"Error querying Snowflake: {str(e)}\n{traceback.format_exc()}"
            print(error_message)
            return {
                "response": error_message,
                "chart": None,
                "sources": []
            }
    
    def _query_snowflake(self, query: str) -> pd.DataFrame:
        """Execute query against Snowflake and return results as DataFrame"""
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA")
        )
        
        try:
            # Check if table exists first
            cursor = conn.cursor()
            cursor.execute(f"SHOW TABLES LIKE '{self.table_name}'")
            tables = cursor.fetchall()
            
            if not tables:
                # For demonstration without proper Snowflake setup, use CSV instead
                print(f"Table {self.table_name} not found. Using fallback CSV data.")
                csv_path = os.getenv("NVDA_CSV_PATH", "/app/NVDA_5yr_history_20250407.csv")
                if os.path.exists(csv_path):
                    return pd.read_csv(csv_path)
                else:
                    raise ValueError(f"Neither Snowflake table nor CSV file found at {csv_path}")
            
            # Execute the actual query
            df = pd.read_sql(query, conn)
            return df
        finally:
            conn.close()
    
    def _generate_chart(self, df, metric="CLOSE") -> str:
        """Generate chart for visualization"""
        # Based on the screenshot, we see DATE, CLOSE, HIGH, LOW, etc.
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            date_col = 'DATE'
        else:
            # Find a date column
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                df[date_col] = pd.to_datetime(df[date_col])
            else:
                return None
        
        # Find the metric column (case insensitive)
        metric_col = None
        for col in df.columns:
            if col.upper() == metric.upper():
                metric_col = col
                break
        
        if not metric_col:
            # Based on screenshot, default to CLOSE if metric not found
            if 'CLOSE' in df.columns:
                metric_col = 'CLOSE'
            else:
                return None
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Build the figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[date_col], df[metric_col], marker="o", linewidth=2, color="#007acc")

        # Title and axes
        ax.set_title(f"NVIDIA {metric_col} Stock Price Over Time", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(f"{metric_col} Price ($)", fontsize=12)
        plt.xticks(rotation=45)
        ax.grid(True, linestyle="--", alpha=0.6)

        # Format y-axis with dollar formatting
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.2f}"))

        # Add data labels for some points (not all to avoid overcrowding)
        # Take every nth point
        n = max(1, len(df) // 10)  # Show about 10 labels
        for i, row in df.iloc[::n].iterrows():
            ax.annotate(
                f'${row[metric_col]:.2f}',
                (row[date_col], row[metric_col]),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=8,
                color='gray'
            )

        # Save the chart to the shared directory
        chart_filename = f"nvda_{metric_col.lower()}_chart.png"
        chart_path = os.path.join(self.charts_dir, chart_filename)
        
        plt.tight_layout()
        try:
            plt.savefig(chart_path)
            print(f"Chart saved to: {chart_path}")
            plt.close()
            
            # Test if file exists and is readable
            if os.path.exists(chart_path):
                return chart_path
            else:
                print(f"Chart file not found at {chart_path} after saving")
                return None
        except Exception as e:
            print(f"Error saving chart: {str(e)}")
            plt.close()
            return None
        
    def _generate_analysis(self, df: pd.DataFrame, query_text: str, 
                          years: Optional[List[int]] = None, quarters: Optional[List[int]] = None) -> str:
        """Generate analysis of financial data using LLM"""
        # Format dataframe as string for context
        df_str = df.head(10).to_string()
        
        # Create a filter description for context
        filter_desc = ""
        if years and len(years) > 0:
            filter_desc += f"Years: {', '.join(map(str, years))}\n"
        if quarters and len(quarters) > 0:
            filter_desc += f"Quarters: {', '.join([f'Q{q}' for q in quarters])}\n"
        
        # Create prompt for analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a financial analyst specializing in NVIDIA.
            Analyze the provided stock data table to answer the user's query.
            Focus on key metrics, trends, and notable changes in stock prices.
            Provide insights in a clear, concise manner suitable for financial analysis.
            Pay special attention to the CLOSE, HIGH, LOW, OPEN prices and trading VOLUME.
            If calculating returns or growth, be explicit about the time periods.
            """),
            ("human", """
            NVIDIA Stock Data:
            {data}
            
            {filter_desc}
            
            User Query: {query}
            
            Provide a detailed analysis based on this stock data.
            """)
        ])
        
        # Generate analysis
        chain = prompt | self.llm
        response = chain.invoke({
            "data": df_str,
            "filter_desc": filter_desc,
            "query": query_text
        })
        
        return response.content