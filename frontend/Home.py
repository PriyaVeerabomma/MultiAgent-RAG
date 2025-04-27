# Home.py
import streamlit as st
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
GITHUB_URL = os.getenv(
    "GITHUB_URL",
    "https://github.com/PriyaVeerabomma/MultiAgent-RAG.git"
)
DEMO_VIDEO_URL = os.getenv(
    "DEMO_VIDEO_URL",
    "https://drive.google.com/drive/folders/19EFzdcQYMKe4BoG-hgAOawTClHCNeqy6?usp=sharing"
)

# Configure Streamlit page
st.set_page_config(
    page_title="Home – NVIDIA Research Assistant",
    page_icon="📊",
    layout="wide"
)

# --- Responsive CSS for both light & dark modes + tighter card dimensions ---
st.markdown("""
<style>
:root {
  --accent: #16a34a;
  --bg-light: #f7fafc;
  --bg-dark: #121212;
  --text-light: #1f2937;
  --text-dark: #e0e0e0;
  --card-light: #ffffff;
  --card-dark: #1e1e1e;
}

/* Page background & text */
@media (prefers-color-scheme: light) {
  .stApp { background-color: var(--bg-light) !important; color: var(--text-light) !important; }
}
@media (prefers-color-scheme: dark) {
  .stApp { background-color: var(--bg-dark) !important; color: var(--text-dark) !important; }
}

/* Link buttons */
.link-btn {
  background-color: var(--accent) !important;
  color: #ffffff !important;
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  font-weight: 500;
  text-decoration: none;
  margin-right: 0.5rem;
  display: inline-block;
}
.link-btn:hover { opacity: 0.85; }

/* Hero */
.hero { text-align: center; padding: 3rem 1rem; }
.hero h1 { font-size: 2.8rem; margin-bottom: 0.3rem; }
.hero p { font-size: 1.2rem; margin-bottom: 2rem; color: rgba(0,0,0,0.6); }
@media (prefers-color-scheme: dark) { .hero p { color: rgba(255,255,255,0.6); } }

/* Card wrapper for gutters */
.card-wrapper { padding: 0.5rem; }

/* Card itself */
.card {
  border-radius: 12px;
  padding: 1rem;
  min-height: 140px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
@media (prefers-color-scheme: light) {
  .card {
    background-color: var(--card-light);
    color: var(--text-light);
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
  }
}
@media (prefers-color-scheme: dark) {
  .card {
    background-color: var(--card-dark);
    color: var(--text-dark);
    box-shadow: 0 2px 6px rgba(0,0,0,0.5);
  }
}

/* Accent titles */
.agent-title {
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: var(--accent);
}

/* Step list */
.step {
  margin-bottom: 0.5rem;
  color: inherit;
}
</style>
""", unsafe_allow_html=True)

# --- Top Links ---
st.markdown(f"""
<div style="display:flex; justify-content:flex-end; margin-top:1rem; margin-bottom:2rem;">
  <a class="link-btn" href="{GITHUB_URL}" target="_blank">View on GitHub</a>
  <a class="link-btn" href="{DEMO_VIDEO_URL}" target="_blank">Watch Demo</a>
</div>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
  <h1>NVIDIA Research Assistant</h1>
  <p>Unlock deep insights into NVIDIA’s business with our multi-agent RAG framework.</p>
</div>
""", unsafe_allow_html=True)

# --- Architecture & Agents ---
st.header("Architecture & Agents")
cols = st.columns(3, gap="large")
descs = [
    ("RAG Agent",
     "Embeds your query, searches NVIDIA’s reports via Pinecone, and synthesizes a historical narrative."),
    ("Snowflake Agent",
     "Generates SQL for your query, retrieves financial data from Snowflake, and produces charts with summaries."),
    ("Web-Search (Tavily) Agent",
     "Performs real-time Tavily web searches, filters top news, and summarizes the latest market intelligence.")
]

for col, (title, text) in zip(cols, descs):
    col.markdown(f"""
      <div class="card-wrapper">
        <div class="card">
          <div class="agent-title">{title}</div>
          <div>{text}</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

# --- Getting Started ---
st.header("Getting Started")
st.markdown("""
<div class="card-wrapper" style="width:100%;">
  <div class="card">
    <div class="step">1. Click **Research** in the sidebar.</div>
    <div class="step">2. Enter your **Research Question**.</div>
    <div class="step">3. (Optional) Filter by **Years** & **Quarters**.</div>
    <div class="step">4. Select one or more **Agents**.</div>
    <div class="step">5. Click **Generate Report** and review the insights.</div>
  </div>
</div>
""", unsafe_allow_html=True)
