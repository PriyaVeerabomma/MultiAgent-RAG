#!/usr/bin/env python
import os
import sys
import subprocess

def main():
    # Get the absolute path to the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up environment
    os.chdir(backend_dir)
    
    # Start uvicorn server
    subprocess.run([
        "uvicorn", "app:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])

if __name__ == "__main__":
    main()
