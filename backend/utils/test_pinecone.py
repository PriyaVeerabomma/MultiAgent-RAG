#!/usr/bin/env python3
"""
Pinecone Connection Tester
-------------------------
Tests the connection to Pinecone using environment variables from the .env file.
Loads environment variables directly from the file to avoid any issues with
dotenv or other loading mechanisms.
"""

from pinecone import Pinecone
import os
import sys
import pprint

def load_env_variables():
    """Load environment variables directly from .env file."""
    # Find the .env file
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(backend_dir, '.env')
    print(f"Reading .env file directly from: {env_file}")

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
                        # Also set in environment
                        os.environ[key] = value
    except Exception as e:
        print(f"Error reading .env file: {e}")
        sys.exit(1)
    
    return env_vars

def test_pinecone_connection(env_vars):
    """Test connection to Pinecone using the provided environment variables."""
    # Get the environment variables from our manually parsed env_vars
    api_key = env_vars.get("PINECONE_API_KEY")
    env = env_vars.get("PINECONE_ENVIRONMENT")
    index_name = env_vars.get("PINECONE_INDEX")

    # Print first few characters of API key for debugging (securely)
    if api_key:
        masked_key = api_key[:5] + '...' + api_key[-5:] if len(api_key) > 10 else "[TOO SHORT]"
        print(f"API KEY: {masked_key} (length: {len(api_key)})")
    else:
        print("API KEY: Not found")
    
    print(f"ENVIRONMENT: {env}")
    print(f"INDEX: {index_name}")

    # Ensure we have all required values
    if not api_key:
        print("Error: PINECONE_API_KEY environment variable not set or empty")
        return False
    
    if not index_name:
        print("Error: PINECONE_INDEX environment variable not set or empty")
        return False

    print("Connecting to Pinecone...")
    try:
        # Initialize Pinecone client with the correct API key
        pc = Pinecone(api_key=api_key)
        
        # List available indexes
        indexes = pc.list_indexes()
        print(f"Available Pinecone indexes: {indexes}")
        
        # Check if our index exists
        index_names = [idx.name for idx in indexes]
        if index_name not in index_names:
            print(f"Warning: Index '{index_name}' not found in available indexes: {index_names}")
            return False

        print(f"Connecting to index '{index_name}'...")
        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        print("Index statistics:")
        print(f"- Dimension: {stats.dimension if hasattr(stats, 'dimension') else 'N/A'}")
        print(f"- Total vector count: {stats.total_vector_count if hasattr(stats, 'total_vector_count') else 0}")
        print(f"- Namespaces: {stats.namespaces if hasattr(stats, 'namespaces') else {}}")
        print(f"- Full stats: {pprint.pformat(stats)}")
        
        print("Pinecone connection successful!")
        return True

    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return False

if __name__ == "__main__":
    # Load environment variables
    env_vars = load_env_variables()
    
    # Test Pinecone connection
    test_pinecone_connection(env_vars)
