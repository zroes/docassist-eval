import json
import os
import chromadb

def get_chroma_collection():
    """
    Connects to the local persistent ChromaDB instance and retrieves the target collection.
    
    Using get_collection() instead of get_or_create_collection() acts as a safety mechanism.
    In a production retrieval setting, if the database doesn't exist, we want the application 
    to throw an explicit error rather than silently creating an empty database.
    """
    # Instantiate the persistent client pointing to our generated data directory
    client = chromadb.PersistentClient(path="../chroma_db")
    
    # Retrieve the collection created during the indexing phase
    collection = client.get_collection(name="watsonx_docs")
    return collection

def retrieve_chunks(collection, queries, top_k=3):
    """
    Executes a batch semantic search against the vector database.
    
    Args:
        collection: The initialized ChromaDB collection object.
        queries (list): A list of query strings to embed and search for.
        top_k (int): The number of most relevant chunks to return per query.
        
    Returns:
        dict: The raw, nested dictionary response from ChromaDB containing ids, distances, metadatas, and documents.
    """
    # We pass the list of queries directly. ChromaDB is highly optimized for batch processing.
    # It will automatically embed the list of queries using the default all-MiniLM-L6-v2 model 
    # and perform parallel approximate nearest neighbor (ANN) searches.
    results = collection.query(query_texts=queries, n_results=top_k)
    return results

def format_retrieval_results(queries, retrieved_data):
    """
    Unpacks the nested dictionary structure returned by ChromaDB into a clean, 
    list-of-dictionaries format suitable for JSON serialization and downstream LLM injection.
    
    Args:
        queries (list): The original list of query strings.
        retrieved_data (dict): The raw output from retrieve_chunks().
        
    Returns:
        list: A structured list containing the queries mapped to their respective formatted chunks.
    """
    formatted_results =[]
    
    # enumerate() gives us both the index (q_idx) and the actual query string.
    # This index is crucial because ChromaDB returns 'lists of lists', where the outer
    # list index perfectly mirrors the original query's position in the batch.
    for q_idx, query in enumerate(queries):
        formatted_chunks =[]
        
        # Loop through the top_k results for this specific query
        for i in range(len(retrieved_data['ids'][q_idx])):
            
            # Extract all relevant metadata and content for a single retrieved chunk
            chunk_info = {
                "chunk_id": retrieved_data['ids'][q_idx][i],             # Unique identifier for tracking/debugging
                "text": retrieved_data['documents'][q_idx][i],           # The actual text payload to pass to the LLM
                "title": retrieved_data['metadatas'][q_idx][i]['title'], # Document title to provide context
                "source": retrieved_data['metadatas'][q_idx][i]['source'], # URL for citations, governance, and explainability
                "category": retrieved_data['metadatas'][q_idx][i]['category'], # High-level grouping
                "distance": retrieved_data['distances'][q_idx][i]        # Similarity score indicating semantic proximity
            }
            formatted_chunks.append(chunk_info)
        
        # Map the original user query to its successfully unpacked chunks
        formatted_results.append({
            "query": query,
            "top_chunks": formatted_chunks
        })
        
    return formatted_results

def run_sample_queries():
    """
    Testing module to simulate user queries, execute the retrieval pipeline, 
    and save the output to a file for visual verification.
    """
    # Initialize the database connection
    collection = get_chroma_collection()
    
    # Define a batch of test queries
    sample_queries =[
        "How do I use Python to chat with a model?",
        "What is RAG useful for?",
        "From a brand new machine, walk me through getting watsonx set up for a basic conversation"
    ]
    
    # Perform the retrieval (Database/Network call)
    retrieved_data = retrieve_chunks(collection, sample_queries, top_k=3)

    # Transform the raw output into our clean, application-ready structure
    results_to_save = format_retrieval_results(sample_queries, retrieved_data)
    
    # Serialize the formatted data to a JSON file
    output_file = "../results/retrieval_examples.json"
    with open(output_file, "w", encoding="utf-8") as out_f:
        # json.dump directly writes the Python dictionary to the file, 
        json.dump(results_to_save, out_f, indent=4)
        
    print(f"\nSuccessfully saved {len(sample_queries)} query examples to results/retrieval_examples.json")

if __name__ == "__main__":
    # Ensure the target directory exists before attempting to write to it.
    # exist_ok=True prevents the script from crashing if the folder is already there.
    os.makedirs("../results", exist_ok=True)
    
    # Execute the test sequence
    run_sample_queries()