# frontend/app.py
import streamlit as st
import requests
import json
import os
from typing import List, Dict, Any

# API endpoint
API_URL = os.environ.get("API_URL", "http://backend:8000")  # When running in Docker

st.title("NVIDIA Research Assistant")
st.write("Ask questions about NVIDIA's financial performance and get comprehensive research insights.")

# User input
query = st.text_input("Research Question", "What was NVIDIA's revenue growth in the past year?")

# Filtering options - modified to allow multiple selections
col1, col2 = st.columns(2)
with col1:
    years = st.multiselect("Years", [2020, 2021, 2022, 2023, 2024], default=[])
with col2:
    quarters = st.multiselect("Quarters", [1, 2, 3, 4], default=[])

# Agent selection
st.write("Select Research Agents:")
col1, col2, col3 = st.columns(3)
with col1:
    use_rag = st.checkbox("Historical Data (RAG)", value=True)
with col2:
    use_snowflake = st.checkbox("Financial Metrics", value=True)
with col3:
    use_websearch = st.checkbox("Latest News & Trends", value=True)

# Generate report button
if st.button("Generate Research Report"):
    agents = []
    if use_rag:
        agents.append("rag")
    if use_snowflake:
        agents.append("snowflake")
    if use_websearch:
        agents.append("websearch")
        
    # Prepare request - modified to handle multiple years and quarters
    payload = {
        "query": query,
        "years": years if years else None,
        "quarters": quarters if quarters else None,
        "agents": agents
    }
    st.write(f"Sending query with parameters: Years={years}, Quarters={quarters}")
    
    with st.spinner("Generating research report..."):
        try:
            response = requests.post(f"{API_URL}/research", json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Display report sections
            st.subheader("Research Report")
            
            if "content" in result:
                st.markdown("### Comprehensive Analysis")
                st.write(result["content"])
            
            if "historical_data" in result and use_rag:
                st.markdown("### Historical Performance")
                st.write(result["historical_data"]["content"])
                
            if "financial_metrics" in result and use_snowflake:
                st.markdown("### Financial Metrics")
                st.write(result["financial_metrics"]["content"])
                
                # Display chart if available with better error handling
                if "chart" in result["financial_metrics"] and result["financial_metrics"]["chart"]:
                    chart_path = result["financial_metrics"]["chart"]
                    try:
                        # Try to display the image
                        st.image(chart_path)
                    except Exception as e:
                        st.error(f"Error displaying chart: {str(e)}")
                        st.write(f"Chart path was: {chart_path}")
                    
            if "latest_insights" in result and use_websearch:
                st.markdown("### Latest Insights & Trends")
                st.write(result["latest_insights"]["content"])
                
                # Display sources if available
                if "sources" in result["latest_insights"] and result["latest_insights"]["sources"]:
                    st.markdown("#### Sources")
                    for source in result["latest_insights"]["sources"]:
                        st.write(source)
                
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")