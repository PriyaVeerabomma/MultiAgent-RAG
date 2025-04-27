# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langraph.orchestrator import ResearchOrchestrator
import os
from dotenv import load_dotenv

# Load environment variables from root first, then backend
backend_dir = os.path.dirname(os.path.abspath(__file__))
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

# Verify key environment variables
tavily_api_key = os.getenv("TAVILY_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

print(f"Environment variables loaded:")
print(f"- TAVILY_API_KEY: {'Set' if tavily_api_key else 'NOT SET'}")
print(f"- OPENAI_API_KEY: {'Set' if openai_api_key else 'NOT SET'}")
print(f"- PINECONE_API_KEY: {'Set' if pinecone_api_key else 'NOT SET'}")

app = FastAPI(title="NVIDIA Research Assistant API")

class ResearchQuery(BaseModel):
    query: str
    years: Optional[List[int]] = None
    quarters: Optional[List[int]] = None
    agents: List[str] = ["rag", "snowflake", "websearch"]

@app.post("/research")
async def generate_research(request: ResearchQuery):
    try:
        # Initialize orchestrator with selected agents
        orchestrator = ResearchOrchestrator(
            use_rag="rag" in request.agents,
            use_snowflake="snowflake" in request.agents,
            use_websearch="websearch" in request.agents
        )
        
        # Generate research report
        print(f"Running orchestrator with query: {request.query}, years: {request.years}, quarters: {request.quarters}")
        result = orchestrator.run(
            query=request.query,
            years=request.years,
            quarters=request.quarters
        )
        
        # Maintain backwards compatibility with the frontend
        if "summary" in result and "content" not in result:
            result["content"] = result["summary"]
        
        return result
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in research endpoint: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

