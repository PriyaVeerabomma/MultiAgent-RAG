# MultiAgent-RAG Backend

This directory contains the backend services for the MultiAgent-RAG system, including document ingestion, vector search, and API endpoints.

## Environment Setup

The system relies on environment variables defined in a `.env` file. Make sure this file exists in the `/backend` directory with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX=your_pinecone_index
```

## Diagnostic and Fixing Tools

The following scripts help diagnose and fix environment issues:

1. **check_pinecone.py** - Verifies Pinecone connection and environment setup
2. **diagnose_env.py** - Comprehensive diagnosis of environment variables and loading
3. **run_ingest.py** - Wrapper script to ensure proper environment before ingestion

## Running the System

### Step 1: Verify Environment Setup

First, run the diagnostic tool to verify your environment setup:

```bash
python diagnose_env.py
```

### Step 2: Check Pinecone Connection

To verify that your Pinecone credentials are working correctly:

```bash
python check_pinecone.py
```

This will test the connection to Pinecone and display information about your index.

### Step 3: Ingest Documents

To ingest documents into Pinecone:

```bash
python run_ingest.py
```

This script loads the environment correctly before running the document ingestion process.

### Step 4: Start the Backend Server

To start the backend API server:

```bash
python start_backend.py
```

## Troubleshooting

If you encounter "Invalid API Key" errors or other Pinecone connection issues:

1. Ensure your `.env` file has the correct Pinecone API key format (should start with `pcsk_`)
2. Verify that your Pinecone environment and index name are correct
3. Check for placeholder values in your environment variables
4. Use the diagnostic tools to identify any loading issues
5. Try running the fixed scripts that load environment variables directly

## Files and Structure

- **utils/** - Directory containing utility scripts for data ingestion and testing
  - **ingest_documents.py** - Script to ingest documents into Pinecone
  - **test_pinecone.py** - Script to test Pinecone connection
  - **direct_test.py** - Alternative script for testing Pinecone connection
  
- **data/** - Directory containing data files to be ingested
  - **nvidia_reports/** - NVIDIA quarterly reports for ingestion

- **check_pinecone.py** - Comprehensive Pinecone connection checker
- **diagnose_env.py** - Environment variables diagnostic tool
- **run_ingest.py** - Wrapper script for document ingestion
