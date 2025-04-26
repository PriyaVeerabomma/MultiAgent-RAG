import os
import re
import sys
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import time

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

# Get credentials from our manually parsed env_vars
pinecone_api_key = env_vars.get("PINECONE_API_KEY")
pinecone_index_name = env_vars.get("PINECONE_INDEX")
openai_api_key = env_vars.get("OPENAI_API_KEY")

# Print first few characters of API keys for debugging (securely)
if pinecone_api_key:
    masked_key = pinecone_api_key[:5] + '...' + pinecone_api_key[-5:] if len(pinecone_api_key) > 10 else "[TOO SHORT]"
    print(f"PINECONE API KEY: {masked_key} (length: {len(pinecone_api_key)})")
else:
    print("ERROR: PINECONE_API_KEY not found in .env file!")
    sys.exit(1)

print(f"PINECONE INDEX: {pinecone_index_name}")

# Initialize OpenAI Embeddings - FIX: Change to text-embedding-3-small to match 1536 dimensions
print("Initializing OpenAI Embeddings with text-embedding-3-small model (1536 dimensions)...")
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Changed from text-embedding-3-large to match Pinecone index dimensions
    openai_api_key=openai_api_key
)

# Initialize Pinecone
print("Initializing Pinecone connection...")
pc = Pinecone(api_key=pinecone_api_key)

# List available indexes for verification
indexes = pc.list_indexes()
print(f"Available Pinecone indexes: {indexes}")

# Connect to the index
print(f"Connecting to index '{pinecone_index_name}'...")
index = pc.Index(pinecone_index_name)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def parse_metadata(filename):
    """Extract metadata (year and quarter) from filename"""
    # Example filenames: "NVIDIA_2021_Q1.pdf", "NVIDIA-2022-Q3-Report.pdf"
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
    print(f"Metadata extracted: Year: {metadata['year']}, Quarter: {metadata['quarter']}")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")
    
    # Process each chunk with rate limiting
    chunks_processed = 0
    errors = 0
    for i, chunk in enumerate(chunks):
        try:
            # Create embedding
            embedding = embedding_model.embed_query(chunk)
            
            # Verify embedding dimension
            if len(embedding) != 1536:
                print(f"Warning: Embedding dimension is {len(embedding)}, expected 1536")
                continue
            
            # Prepare metadata for this chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["text"] = chunk
            chunk_metadata["chunk_id"] = str(i)
            
            # Upload to Pinecone
            index.upsert(
                vectors=[(f"{filename.replace('.pdf', '')}-{i}", embedding, chunk_metadata)]
            )
            
            chunks_processed += 1
            
            # Rate limiting to avoid API throttling
            if i % 10 == 0 and i > 0:
                print(f"Uploaded {chunks_processed}/{len(chunks)} chunks from {filename}")
                time.sleep(1)  # Add delay every 10 chunks
                
        except Exception as e:
            errors += 1
            print(f"Error processing chunk {i} from {filename}: {e}")
            if errors >= 5:  # Limit number of errors before giving up
                print(f"Too many errors ({errors}), skipping remaining chunks")
                break
    
    print(f"Completed uploading {chunks_processed} of {len(chunks)} chunks from {filename}")
    return chunks_processed

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

if __name__ == "__main__":
    # Directory containing NVIDIA quarterly reports
    pdf_directory = "./data/nvidia_reports"
    
    # Check if directory exists
    if not os.path.isabs(pdf_directory):
        # Convert to absolute path if it's relative
        pdf_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), pdf_directory)
    
    print(f"Processing PDFs from: {pdf_directory}")
    
    # Process all PDFs in directory
    process_directory(pdf_directory)
