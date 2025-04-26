import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

def test_pinecone_search():
    # Load environment variables
    load_dotenv()
    
    # Get API keys
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not pinecone_api_key or not openai_api_key:
        print("Error: API keys not found in environment variables")
        return
    
    # Initialize OpenAI embeddings
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key
    )
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Connect to index
    index_name = "nvidia-reports"
    print(f"Connecting to index '{index_name}'...")
    index = pc.Index(index_name)
    
    # Get test query from user
    query = input("Enter a query about NVIDIA (e.g., 'What was NVIDIA's revenue growth?'): ")
    
    # Optional: filter by year/quarter
    use_filter = input("Do you want to filter by year/quarter? (y/n): ").lower() == 'y'
    filter_dict = {}
    
    if use_filter:
        year = input("Enter year to filter by (e.g., 2023) or leave empty: ")
        quarter = input("Enter quarter to filter by (1-4) or leave empty: ")
        
        if year:
            filter_dict["year"] = {"$eq": year}
        if quarter:
            filter_dict["quarter"] = {"$eq": f"q{quarter}"}
    
    # Generate embedding for query
    print("Generating embedding for query...")
    query_embedding = embedding_model.embed_query(query)
    
    # Perform search
    print(f"Searching Pinecone index with{' filters' if filter_dict else ' no filters'}...")
    search_results = index.query(
        vector=query_embedding,
        filter=filter_dict if filter_dict else None,
        top_k=5,
        include_metadata=True
    )
    
    # Print results
    print(f"\nSearch Results for: '{query}'")
    print("-" * 50)
    
    if not search_results.matches:
        print("No results found.")
        return
    
    for i, match in enumerate(search_results.matches):
        print(f"Result {i+1} (Score: {match.score:.4f}):")
        print(f"  Source: {match.metadata.get('source', 'Unknown')}")
        print(f"  Year: {match.metadata.get('year', 'Unknown')}")
        print(f"  Quarter: {match.metadata.get('quarter', 'Unknown')}")
        print(f"  Text: {match.metadata.get('text', '')[:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    test_pinecone_search()
