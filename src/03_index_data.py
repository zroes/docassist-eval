import json
import chromadb


def load_chunks(filepath="../data/processed/chunks.jsonl"):
    """
    Loads a JSON Lines file into a list of dictionaries

    Each line is expected to be a JSON object
    """
    chunks =[]
    
    with open(filepath, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    return chunks



def index_chunks_to_chroma(chunks):
    """
    Takes the list of chunk dictionaries and loads them into ChromaDB.
    """
    print("Initializing ChromaDB...")
    # This creates a local folder called 'chroma_db' in the project
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create a collection (think of this like a table in a database)
    collection = client.get_or_create_collection(name="watsonx_docs")
    

    # ChromaDB requires data to be passed as separate lists, not dictionaries!
    ids =[]
    documents = []
    metadatas = []
    
    print("Preparing data for indexing...")

    # Iterates through the list of dictionaries and breaks up the data
    # into specialized lists for chromadb

    for chunk in chunks:
        ids.append(chunk['chunk_id'])
        documents.append(chunk['text'])
        metadata = {
            "title": chunk['title'],
            "category": chunk['category'],
            "source": chunk['source'],
            "doc_id": chunk['doc_id']
        }
        metadatas.append(metadata)


    print(f"Adding {len(documents)} chunks to the Vector Database. This might take a minute...")
    
    # Adds our 3 lists to the chromadb database
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    
    print("Indexing complete!")

if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Found {len(chunks)} chunks")
    # print(f"Sample chunk: {type(chunks[0])}")
    index_chunks_to_chroma(chunks)