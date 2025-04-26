#!/usr/bin/env python3
"""
Run Ingest Script
----------------
This script ensures proper environment variable setup before running the document ingestion process.
It directly reads the .env file and checks for required Pinecone credentials before executing.
It also verifies embedding dimensions match the Pinecone index.
"""

import os
import sys
import subprocess
import json
import re
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def load_env_vars():
    """Load environment variables from .env file."""
    # Get the current directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(backend_dir, '.env')
    print(f"Reading .env file from: {env_file}")

    # Check if .env file exists
    if not os.path.exists(env_file):
        print(f"ERROR: .env file not found at {env_file}")
        sys.exit(1)
    
    # Dictionary to store environment variables
    env_vars = {}

    # Read the .env file manually without any package
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    # Split only on the first equals sign
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        # Remove any quotes
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith('\'') and value.endswith('\'')):
                            value = value[1:-1]
                        env_vars[key] = value
                        # Set environment variable
                        os.environ[key] = value
    except Exception as e:
        print(f"ERROR: Failed to read .env file: {e}")
        sys.exit(1)
    
    return env_vars

def check_required_vars(env_vars):
    """Check required environment variables."""
    # Check required environment variables
    required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not env_vars.get(var)]
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    # Verify Pinecone API key
    pinecone_api_key = env_vars.get("PINECONE_API_KEY")
    if not pinecone_api_key.startswith("pcsk_"):
        print(f"ERROR: Invalid Pinecone API key format. Should start with 'pcsk_'")
        sys.exit(1)
    
    # Print environment variables (masked for security)
    print("\nVerified environment variables:")
    print(f"- PINECONE_API_KEY: {pinecone_api_key[:5]}...{pinecone_api_key[-5:]} (length: {len(pinecone_api_key)})")
    print(f"- PINECONE_INDEX: {env_vars.get('PINECONE_INDEX')}")
    print(f"- OPENAI_API_KEY: {env_vars.get('OPENAI_API_KEY')[:5]}... (masked)")

def check_embedding_model(env_vars):
    """Check the embedding model and verify dimensions match the Pinecone index."""
    print_section("Checking Embedding Model")
    
    try:
        # Get the embedding model from ingest_documents.py
        utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils')
        ingest_file = os.path.join(utils_dir, 'ingest_documents.py')
        
        if not os.path.exists(ingest_file):
            print(f"ERROR: ingest_documents.py not found at {ingest_file}")
            return None, None
        
        with open(ingest_file, 'r') as f:
            content = f.read()
        
        # Extract the embedding model name
        model_match = re.search(r'model="([^"]+)"', content)
        if model_match:
            model_name = model_match.group(1)
            print(f"Embedding model found in ingest_documents.py: {model_name}")
        else:
            print("ERROR: Could not find embedding model name in ingest_documents.py")
            return None, None
        
        # Initialize the embedding model
        embedding_model = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=env_vars.get("OPENAI_API_KEY")
        )
        
        # Get a sample embedding to check dimensions
        test_text = "This is a test text for embedding dimension verification."
        embedding = embedding_model.embed_query(test_text)
        embedding_dim = len(embedding)
        
        print(f"Embedding dimensions: {embedding_dim}")
        
        # Get the Pinecone index dimensions
        pc = Pinecone(api_key=env_vars.get("PINECONE_API_KEY"))
        index_name = env_vars.get("PINECONE_INDEX")
        index_info = pc.describe_index(index_name)
        
        if hasattr(index_info, 'dimension'):
            index_dim = index_info.dimension
        else:
            index_dim = index_info.get('dimension')
        
        print(f"Pinecone index dimensions: {index_dim}")
        
        # Check if dimensions match
        if embedding_dim == index_dim:
            print(f"✓ Dimensions match: Both are {embedding_dim}")
            return model_name, embedding_dim
        else:
            print(f"✗ Dimension mismatch: Embedding model has {embedding_dim} dimensions, but Pinecone index has {index_dim} dimensions")
            print("\nPlease run fix_dimension_mismatch.py to resolve this issue:")
            print("  python fix_dimension_mismatch.py")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR checking embedding model: {e}")
        return None, None

def run_ingestion():
    """Run the document ingestion process."""
    print_section("Running Document Ingestion")
    
    # Run the ingest_documents.py script
    ingest_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils', 'ingest_documents.py')
    
    if not os.path.exists(ingest_script):
        print(f"ERROR: Ingest script not found at {ingest_script}")
        sys.exit(1)
    
    print("Running document ingestion process...")
    print("-"*80)
    
    # Run the ingest script with our environment variables
    try:
        result = subprocess.run([sys.executable, ingest_script], 
                                env=os.environ,
                                check=True)
        print("-"*80)
        print("Document ingestion completed successfully!")
    except subprocess.CalledProcessError as e:
        print("-"*80)
        print(f"ERROR: Document ingestion failed with exit code {e.returncode}")
        
        # Check if the error might be due to dimension mismatch
        if e.returncode != 0:
            output = e.output.decode('utf-8') if hasattr(e, 'output') and e.output else ""
            if "dimension" in output.lower() and "does not match" in output.lower():
                print("\nDimension mismatch detected. Please run fix_dimension_mismatch.py to resolve this issue:")
                print("  python fix_dimension_mismatch.py")
            
        sys.exit(e.returncode)

def main():
    print_section("DOCUMENT INGESTION RUNNER")
    
    # Load environment variables
    env_vars = load_env_vars()
    
    # Check required variables
    check_required_vars(env_vars)
    
    # Check embedding model and dimensions
    model_name, embedding_dim = check_embedding_model(env_vars)
    
    # Run ingestion
    run_ingestion()

if __name__ == "__main__":
    main()
