import json
import chromadb

def get_chroma_collection():
    """
    TODO 1: Connect to the existing PersistentClient at "./chroma_db"
    TODO 2: Get the "watsonx_docs" collection and return it.
    """
    # Your code here
    pass

def retrieve_chunks(collection, query, top_k=3):
    """
    TODO 3: Use collection.query() to search for the `query` text.
    TODO 4: Request `n_results=top_k`.
    TODO 5: Return the results.
    """
    # Your code here
    pass

def run_sample_queries():
    collection = get_chroma_collection()
    
    # Write 3 to 5 questions related to the watsonx docs you downloaded!
    sample_queries =[
        "How do I use the Python SDK to chat with a model?",
        "What is the flight distance between Paris and Bangalore?",
        # Add a couple more based on the pages you scraped!
    ]
    
    results_to_save =[]
    
    for query in sample_queries:
        print(f"\nQuestion: {query}")
        # Call your retrieval function
        retrieved_data = retrieve_chunks(collection, query, top_k=3)
        
        # TODO 6: Extract the document text and titles from `retrieved_data` 
        # (ChromaDB returns a slightly complex dictionary, you'll need to inspect it!)
        
        # Save the results in a nice format
        results_to_save.append({
            "query": query,
            "top_chunks": retrieved_data # Feel free to format this nicely
        })
    
    # TODO 7: Save `results_to_save` to "results/retrieval_examples.json"
    print("\nSaved examples to results/retrieval_examples.json")

if __name__ == "__main__":
    # Make sure the results directory exists
    import os
    os.makedirs("results", exist_ok=True)
    run_sample_queries()