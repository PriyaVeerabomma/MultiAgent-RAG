# pages/Research.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables from root first, then frontend
frontend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(frontend_dir)

# Check if .env exists in root directory first
root_env_file = os.path.join(root_dir, '.env')
frontend_env_file = os.path.join(frontend_dir, '.env')

if os.path.exists(root_env_file):
    load_dotenv(root_env_file)
    print(f"Using .env file from root directory: {root_env_file}")
elif os.path.exists(frontend_env_file):
    load_dotenv(frontend_env_file)
    print(f"Using .env file from frontend directory: {frontend_env_file}")
else:
    load_dotenv()  # Fall back to default behavior

# Determine appropriate API URL based on environment
# First try CONTAINER_API_URL (for Docker container communication)
API_URL = os.getenv("CONTAINER_API_URL", os.getenv("API_URL", "http://localhost:8000"))
print(f"Using API URL: {API_URL}")

# Page config
st.set_page_config(
    page_title="Research - NVIDIA Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.section-header { color: #4f46e5; font-weight: 600; margin-top: 1rem; }
.stButton button { background-color: #4f46e5 !important; color: white !important; width: 100%; }
</style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("Research Filters & Sources")
query = st.sidebar.text_input("Enter Research Question:")
years = st.sidebar.multiselect("Years", options=list(range(2020, 2026)))
quarters = st.sidebar.multiselect("Quarters", options=[1, 2, 3, 4])
st.sidebar.subheader("Agents")
use_rag = st.sidebar.checkbox("Historical Data (RAG Agent)", True)
use_snowflake = st.sidebar.checkbox("Financial Analysis (Snowflake Agent)", True)
use_websearch = st.sidebar.checkbox("Market Intelligence (Websearch Agent)", True)
run = st.sidebar.button("Generate Report")

# Main area
st.title("NVIDIA Research Assistant - Research")
if not run:
    st.info("Use the sidebar to enter your question, filters, and agents, then click 'Generate Report'.")
else:
    if not query:
        st.error("Please input a research question.")
    else:
        agents = [
            a for a, flag in [
                ("rag", use_rag),
                ("snowflake", use_snowflake),
                ("websearch", use_websearch)
            ] if flag
        ]
        if not agents:
            st.error("Select at least one research agent.")
        else:
            payload = {
                "query": query,
                "years": years or None,
                "quarters": quarters or None,
                "agents": agents
            }
            with st.spinner("Generating report..."):
                try:
                    res = requests.post(f"{API_URL}/research", json=payload)
                    res.raise_for_status()
                    data = res.json()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            # Display Report
            st.header("Comprehensive Research Report")
            st.markdown(data.get("content", "No report available."))

            # Extended details
            sections = []
            if use_rag and data.get("historical_data"):
                sections.append(("Historical Analysis", data["historical_data"]))
            if use_snowflake and data.get("financial_metrics"):
                sections.append(("Financial Analysis", data["financial_metrics"]))
            if use_websearch and data.get("latest_insights"):
                sections.append(("Market Intelligence", data["latest_insights"]))

            if sections:
                with st.expander("Extended Research & Detailed Analysis"):
                    tabs = st.tabs([name for name, _ in sections])
                    for i, (name, section) in enumerate(sections):
                        with tabs[i]:
                            st.markdown(f"<div class='section-header'>{name}</div>", unsafe_allow_html=True)
                            st.markdown(section.get("content", ""))
                            if section.get("sources"):
                                st.write("**Sources:**")
                                for src in section["sources"]:
                                    st.write(f"- {src}")
