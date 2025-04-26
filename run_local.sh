#!/bin/bash
# Script to run the MultiAgent-RAG application locally

# Function to stop all processes on exit
cleanup() {
  echo "Stopping all processes..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  exit
}

# Set up trap to call cleanup function on exit
trap cleanup INT TERM

# Print a header
echo "========================================================"
echo "        NVIDIA Research Assistant - Local Setup         "
echo "========================================================"

# Check environment variables
MISSING_VARS=0
ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
  echo "❌ .env file not found!"
  MISSING_VARS=1
else
  echo "✅ .env file found"
  
  # Check required API keys
  source $ENV_FILE
  
  if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set in .env file"
    MISSING_VARS=1
  else
    echo "✅ OPENAI_API_KEY is set"
  fi
  
  if [ -z "$TAVILY_API_KEY" ]; then
    echo "❌ TAVILY_API_KEY not set in .env file"
    MISSING_VARS=1
  else
    echo "✅ TAVILY_API_KEY is set"
  fi
  
  if [ -z "$PINECONE_API_KEY" ]; then
    echo "❌ PINECONE_API_KEY not set in .env file"
    MISSING_VARS=1
  else
    echo "✅ PINECONE_API_KEY is set"
  fi
fi

if [ $MISSING_VARS -eq 1 ]; then
  echo "Please fix the missing environment variables and try again."
  exit 1
fi

echo "========================================================"

# Start backend server
echo "Starting backend server..."
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Start frontend
echo "Starting frontend..."
cd frontend
streamlit run app.py &
FRONTEND_PID=$!
cd ..

echo "========================================================"
echo "Application started!"
echo "Backend API:  http://localhost:8000"
echo "Frontend UI:  http://localhost:8501"
echo "========================================================"
echo "Press Ctrl+C to stop all services"

# Wait for processes to finish
wait $BACKEND_PID $FRONTEND_PID
