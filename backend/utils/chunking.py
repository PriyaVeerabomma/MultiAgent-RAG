import os
import pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import re

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
index = pinecone.Index("nvidia-reports")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to parse year and quarter from filename or content
def parse_metadata(filename, content):
    # Example: Extract year and quarter from filename like "NVIDIA_2023_Q1.pdf"
    year_match = re.search(r'20(\d{2})', filename)
    quarter_match = re.search(r'Q([1-4])', filename)
    
    year = year_match.group(0) if year_match else "Unknown"
    quarter = f"q{quarter_match.group(1)}" if quarter_match else "Unknown"
    
    return {
        "source": filename,
        "year": year,
        "quarter": quarter
    }

# Process and upload reports
def process_reports(directory):
    # Text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Process each PDF in directory
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            print(f"Processing {filepath}...")
            
            # Extract text
            text = extract_text_from_pdf(filepath)
            
            # Get metadata
            metadata = parse_metadata(filename, text)
            
            # Split text into chunks
            chunks = text_splitter.split_text(text)
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Create embedding
                embedding = embedding_model.embed_query(chunk)
                
                # Prepare metadata for this chunk
                chunk_metadata = metadata.copy()
                chunk_metadata["text"] = chunk
                chunk_metadata["chunk_id"] = i
                
                # Upload to Pinecone
                index.upsert(
                    vectors=[(f"{filename}-{i}", embedding, chunk_metadata)]
                )
            
            print(f"Uploaded {len(chunks)} chunks from {filename}")

# Run the processor
pdf_directory = "./data/nvidia_reports"
process_reports(pdf_directory)