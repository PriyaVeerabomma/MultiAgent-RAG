#!/usr/bin/env python
import os
import re
import sys
import time
import PyPDF2
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# Backend directory path
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
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    env_vars[key] = value
                    # Also set in environment
                    os.environ[key] = value
except Exception as e:
    print(f"Error reading .env file: {e}")
    sys.exit(1)

# Extract Pinecone and OpenAI credentials
pinecone_api_key = env_vars.get('PINECONE_API_KEY')
openai_api_key = env_vars.get('OPENAI_API_KEY')
pinecone_index = env_vars.get('PINECONE_INDEX', 'nvidia-reports')

# Print information (securely masked)
if pinecone_api_key:
    masked_key = pinecone_api_key[:5] + '...' + pinecone_api_key[-5:] if len(pinecone_api_key) > 10 else "[TOO SHORT]"
    print(f"Using PINECONE_API_KEY: {masked_key} (length: {len(pinecone_api_key)})")
else:
    print("ERROR: PINECONE_API_KEY not found in .env file!")
    sys.exit(1)

if openai_api_key:
    masked_key = openai_api_key[:5] + '...' + openai_api_key[-5:] if len(openai_api_key) > 10 else "[TOO SHORT]"
    print(f"Using OPENAI_API_KEY: {masked_key} (length: {len(openai_api_key)})")
else:
    print("ERROR: OPENAI_API_KEY not found in .env file!")
    sys.exit(1)

print(f"Using PINECONE_INDEX: {pinecone_index}")

# Initialize OpenAI Embeddings
print("Initializing OpenAI embeddings...")
try:
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key
    )
    print("Successfully initialized OpenAI embeddings")
except Exception as e:
    print(f"Failed to initialize OpenAI embeddings: {str(e)}")
    sys.exit(1)

# Initialize Pinecone
print("Initializing Pinecone...")
try:
    pc = Pinecone(api_key=pinecone_api_key)
    print(f"Successfully connected to Pinecone")
    
    # List available indexes
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]
    print(f"Available Pinecone indexes: {index_names}")
    
    if pinecone_index not in index_names:
        print(f"WARNING: Index '{pinecone_index}' not found!")
        sys.exit(1)
        
    # Connect to the index
    index = pc.Index(pinecone_index)
    print(f"Successfully connected to index '{pinecone_index}'")
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"Current index stats: {stats}")
    
except Exception as e:
    print(f"Error with Pinecone initialization: {str(e)}")
    sys.exit(1)

# Helper functions
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def parse_metadata(filename):
    """Extract metadata (year and quarter) from filename"""
    # Example filenames: "NVIDIA_2023_Q1.pdf", "NVIDIA-2022-Q3-Report.pdf"
    year_match = re.search(r'20(\d{2})', filename)
    quarter_match = re.search(r'[qQ]([1-4])', filename, re.IGNORECASE)
    
    year = year_match.group(0) if year_match else "Unknown"
    quarter = f"q{quarter_match.group(1)}" if quarter_match else "Unknown"
    
    return {
        "source": filename,
        "year": year,
        "quarter": quarter.lower()  # Ensure lowercase for consistent queries
    }

def process_pdf(pdf_path):
    """Process a single PDF document"""
    filename = os.path.basename(pdf_path)
    print(f"Processing {filename}...")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print(f"No text extracted from {filename}, skipping")
        return 0
    
    # Get metadata
    metadata = parse_metadata(filename)
    print(f"Extracted metadata: Year: {metadata['year']}, Quarter: {metadata['quarter']}")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")
    
    # Process each chunk with rate limiting
    vectors_to_upsert = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Create embedding
            embedding = embedding_model.embed_query(chunk)
            
            # Prepare metadata for this chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["text"] = chunk
            chunk_metadata["chunk_id"] = str(i)
            
            # Add to vectors list
            vector_id = f"{filename.replace('.pdf', '')}-{i}"
            vectors_to_upsert.append((vector_id, embedding, chunk_metadata))
            
            # Batch upload every 100 vectors
            if len(vectors_to_upsert) >= 100:
                index.upsert(vectors=vectors_to_upsert)
                print(f"Uploaded batch of {len(vectors_to_upsert)} vectors")
                vectors_to_upsert = []
                time.sleep(1)  # Add delay for rate limiting
                
        except Exception as e:
            print(f"Error processing chunk {i} from {filename}: {e}")
    
    # Upload any remaining vectors
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            print(f"Uploaded final batch of {len(vectors_to_upsert)} vectors")
        except Exception as e:
            print(f"Error uploading final batch: {e}")
    
    print(f"Completed uploading {len(chunks)} chunks from {filename}")
    return len(chunks)

def process_directory(directory_path):
    """Process all PDF files in a directory"""
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    total_chunks = 0
    for filename in pdf_files:
        filepath = os.path.join(directory_path, filename)
        chunks_processed = process_pdf(filepath)
        total_chunks += chunks_processed
    
    print(f"Processing completed. Total chunks uploaded: {total_chunks}")

# Insert a sample document for testing
def create_sample_document():
    print("\nCreating a sample document for testing...")
    try:
        # Sample text about NVIDIA
        sample_text = """
        NVIDIA Corporation (NASDAQ: NVDA) reported strong financial results for the first quarter of 2023.
        Revenue reached $7.19 billion, marking a 46% increase from the previous year.
        The Graphics segment contributed $3.62 billion, while the Compute & Networking segment generated $3.57 billion.
        Gross margin improved to 67.1%, up from 64.8% in the year-ago quarter.
        Gaming revenue was $2.24 billion, and Data Center revenue hit a record $3.75 billion.
        CEO Jensen Huang stated, "We are seeing unprecedented demand for our data center products."
        For the second quarter, NVIDIA expects revenue to be approximately $8.1 billion.
        The company also announced a 4-for-1 stock split effective in June 2023.
        NVIDIA continues to lead in AI and high-performance computing solutions.
        """
        
        # Create metadata
        metadata = {
            "source": "sample_nvidia_2023_q1.txt",
            "year": "2023",
            "quarter": "q1"
        }
        
        # Generate embedding
        embedding = embedding_model.embed_query(sample_text)
        
        # Add metadata
        metadata["text"] = sample_text
        
        # Upload to Pinecone
        index.upsert(
            vectors=[("sample_document_2023_q1", embedding, metadata)]
        )
        
        print("Successfully created sample document!")
        return True
    except Exception as e:
        print(f"Error creating sample document: {e}")
        return False

if __name__ == "__main__":
    # Directory containing NVIDIA quarterly reports
    pdf_directory = os.path.join(backend_dir, "data/nvidia_reports")
    
    print(f"\nProcessing PDFs from: {pdf_directory}")
    
    # Check if directory exists and has files
    if not os.path.exists(pdf_directory):
        print(f"Creating directory: {pdf_directory}")
        os.makedirs(pdf_directory, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found. Creating a sample document instead.")
        create_sample_document()
    else:
        # Process all PDFs in directory
        process_directory(pdf_directory)
        
    # Get final stats
    stats = index.describe_index_stats()
    print(f"\nFinal index stats: {stats}")
    print("\nData ingestion complete!")
