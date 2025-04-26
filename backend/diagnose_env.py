#!/usr/bin/env python3
"""
Environment Variables Diagnostic Tool
------------------------------------
This script diagnoses issues with environment variables for the MultiAgent-RAG system,
particularly focusing on Pinecone credentials.
"""

import os
import sys
import subprocess
import re
import importlib

def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def check_python_environment():
    """Check Python environment and versions."""
    print_section("Python Environment")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check for dotenv
    try:
        import dotenv
        print(f"python-dotenv version: {dotenv.__version__}")
        print(f"python-dotenv location: {dotenv.__file__}")
    except ImportError:
        print("python-dotenv not installed")
    
    # Check for pinecone
    try:
        import pinecone
        print(f"pinecone-client version: {getattr(pinecone, '__version__', 'unknown')}")
        print(f"pinecone-client location: {pinecone.__file__}")
    except ImportError:
        print("pinecone-client not installed")

def check_env_file():
    """Check the .env file contents."""
    print_section("Environment File")
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(backend_dir, '.env')
    
    if not os.path.exists(env_file):
        print(f"ERROR: .env file not found at {env_file}")
        return
    
    print(f"Found .env file at: {env_file}")
    
    # Read and mask sensitive values
    sensitive_keys = ["API_KEY", "PASSWORD", "SECRET"]
    
    try:
        with open(env_file, 'r') as f:
            lines = f.readlines()
            
        print(f"File contains {len(lines)} lines")
        print("\nSensitive values are masked for security:")
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Check if this is a sensitive key
                is_sensitive = any(s in key.upper() for s in sensitive_keys)
                
                if is_sensitive and len(value) > 10:
                    masked_value = value[:5] + '...' + value[-5:]
                    print(f"{key}={masked_value} (length: {len(value)})")
                else:
                    print(f"{key}={value}")
    except Exception as e:
        print(f"Error reading .env file: {e}")

def check_environment_variables():
    """Check environment variables."""
    print_section("Environment Variables")
    
    # Check for placeholder values
    placeholder_patterns = [
        r'your_.*',
        r'YOUR_.*',
        r'example.*',
        r'EXAMPLE.*',
        r'placeholder.*',
        r'PLACEHOLDER.*',
        r'<.*>',
        r'\[.*\]'
    ]
    
    # Check OS environment variables first
    print("OS Environment Variables:")
    pinecone_env_vars = {
        key: value for key, value in os.environ.items()
        if 'PINECONE' in key
    }
    
    if pinecone_env_vars:
        for key, value in pinecone_env_vars.items():
            if key.endswith('API_KEY') and len(value) > 10:
                masked_value = value[:5] + '...' + value[-5:]
                print(f"  {key}={masked_value} (length: {len(value)})")
            else:
                print(f"  {key}={value}")
            
            # Check for placeholder values
            for pattern in placeholder_patterns:
                if re.match(pattern, value, re.IGNORECASE):
                    print(f"  WARNING: {key} contains a placeholder value!")
    else:
        print("  No Pinecone environment variables found in OS environment")
    
    # Now check by loading from .env file directly
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(backend_dir, '.env')
    
    if os.path.exists(env_file):
        print("\nEnvironment variables from .env file:")
        env_vars = {}
        
        # Read the .env file manually
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        # Remove any quotes
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith('\'') and value.endswith('\'')):
                            value = value[1:-1]
                        env_vars[key] = value
        
        # Check Pinecone variables
        for key, value in env_vars.items():
            if 'PINECONE' in key:
                if key.endswith('API_KEY') and len(value) > 10:
                    masked_value = value[:5] + '...' + value[-5:]
                    print(f"  {key}={masked_value} (length: {len(value)})")
                else:
                    print(f"  {key}={value}")
                
                # Check for placeholder values
                for pattern in placeholder_patterns:
                    if re.match(pattern, value, re.IGNORECASE):
                        print(f"  WARNING: {key} contains a placeholder value in .env file!")

def check_pinecone_initialization():
    """Check Pinecone initialization in various files."""
    print_section("Pinecone Initialization Check")
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Files to check
    files_to_check = [
        os.path.join(backend_dir, 'utils', 'ingest_documents.py'),
        os.path.join(backend_dir, 'utils', 'test_pinecone.py'),
        os.path.join(backend_dir, 'utils', 'direct_test.py')
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"\nChecking {os.path.basename(file_path)}:")
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check how Pinecone is initialized
                pinecone_init = re.search(r'(?:pinecone\.init|Pinecone\()(.*?)(?:\)|$)', content, re.DOTALL)
                if pinecone_init:
                    print(f"  Pinecone initialization found: {pinecone_init.group(0).strip()}")
                else:
                    print("  No Pinecone initialization found")
                
                # Check how environment variables are loaded
                dotenv_load = re.search(r'(?:from\s+dotenv\s+import|load_dotenv)(.*?)(?:\n|$)', content, re.DOTALL)
                if dotenv_load:
                    print(f"  Environment loading: {dotenv_load.group(0).strip()}")
                else:
                    print("  No dotenv loading found")
                
                # Check how API key is accessed
                api_key_access = re.search(r'(?:api_key|PINECONE_API_KEY).*?=.*?(os\.(?:environ|getenv)|env_vars)', content)
                if api_key_access:
                    print(f"  API key access method: {api_key_access.group(0).strip()}")
                else:
                    print("  No API key access method found")
                
            except Exception as e:
                print(f"  Error analyzing file: {e}")
        else:
            print(f"\nFile not found: {file_path}")

def test_direct_env_loading():
    """Test loading environment variables directly from the .env file."""
    print_section("Direct Environment Loading Test")
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(backend_dir, '.env')
    
    print(f"Reading .env file directly from: {env_file}")
    
    # Dictionary to store environment variables
    env_vars = {}

    # Read the .env file manually
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        # Remove any quotes
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith('\'') and value.endswith('\'')):
                            value = value[1:-1]
                        env_vars[key] = value
        
        # Check critical Pinecone variables
        api_key = env_vars.get('PINECONE_API_KEY')
        environment = env_vars.get('PINECONE_ENVIRONMENT')
        index = env_vars.get('PINECONE_INDEX')
        
        print("\nChecking critical Pinecone variables:")
        
        if api_key:
            masked_key = api_key[:5] + '...' + api_key[-5:] if len(api_key) > 10 else "[TOO SHORT]"
            print(f"✓ PINECONE_API_KEY is set (length: {len(api_key)})")
            
            # Check if it looks like a valid Pinecone API key
            if api_key.startswith('pcsk_'):
                print(f"✓ PINECONE_API_KEY appears to be valid")
            else:
                print(f"✗ PINECONE_API_KEY does not have the expected format (should start with 'pcsk_')")
        else:
            print("✗ PINECONE_API_KEY is not set")
        
        if environment:
            print(f"✓ PINECONE_ENVIRONMENT is set to: {environment}")
            
            # Check if it looks like a valid environment
            if re.match(r'^[a-z0-9-]+$', environment):
                print(f"✓ PINECONE_ENVIRONMENT appears to be valid")
            else:
                print(f"✗ PINECONE_ENVIRONMENT does not have the expected format")
        else:
            print("✗ PINECONE_ENVIRONMENT is not set")
        
        if index:
            print(f"✓ PINECONE_INDEX is set to: {index}")
            
            # Check if it looks like a valid index name
            if re.match(r'^[a-zA-Z0-9-]+$', index):
                print(f"✓ PINECONE_INDEX appears to be valid")
            else:
                print(f"✗ PINECONE_INDEX does not have the expected format")
        else:
            print("✗ PINECONE_INDEX is not set")
            
    except Exception as e:
        print(f"Error reading .env file: {e}")

def recommend_fixes():
    """Recommend fixes based on the analysis."""
    print_section("Recommendations")
    
    print("""
1. Ensure your .env file has the correct Pinecone credentials:
   - PINECONE_API_KEY should start with 'pcsk_'
   - PINECONE_ENVIRONMENT should be a valid region (e.g., 'us-east-1')
   - PINECONE_INDEX should match your Pinecone index name

2. Use direct file loading in your scripts:
   - Instead of relying on dotenv, read the .env file directly
   - See the updated utils/test_pinecone.py for an example

3. Check for placeholder values:
   - Make sure no 'your_*' or placeholder values remain in your .env file

4. Use the run_ingest.py script:
   - This ensures proper environment setup before running ingestion

5. Verify Pinecone connection:
   - Use check_pinecone.py to verify the connection before ingestion

If problems persist, you can try:
- Reinstalling the Pinecone client: pip install -U pinecone-client
- Creating a new Pinecone API key in the Pinecone console
- Checking your Pinecone index name in the Pinecone console
    """)

def main():
    """Main function to run all checks."""
    print("MultiAgent-RAG Environment Diagnostics")
    print("Version 1.0.0")
    print("="*80)
    
    # Run all diagnostic checks
    check_python_environment()
    check_env_file()
    check_environment_variables()
    check_pinecone_initialization()
    test_direct_env_loading()
    recommend_fixes()
    
    print("\nDiagnostic complete. Please review the output above for any issues.")
    print("="*80)

if __name__ == "__main__":
    main()
