import os
import glob
import json

def chunk_text(text, chunk_size=300, overlap=50):
    """
    Takes a single string of text and splits it into overlapping chunks.
    """
    words = text.split() 
    
    # Calculate the stride to ensure the requested overlap between chunks
    step = chunk_size - overlap
    
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        # Yielding instead of returning a massive list keeps memory footprint minimal
        yield " ".join(chunk)


def process_documents():
    """
    Reads raw structured JSON documents, breaks their text into semantic chunks, 
    adds metadata, and saves them to a streaming-friendly .jsonl file.
    """
    raw_json_files = glob.glob("../data/raw/json/*.json")
    output_file = "../data/processed/chunks.jsonl"

    total_chunks_created = 0
    print(f"Found {len(raw_json_files)} JSON files to process.")

    # Open the output file once in write mode. 

    with open(output_file, "w", encoding="utf-8") as out_f:

        for file_path in raw_json_files:
            with open(file_path, "r", encoding="utf-8") as in_f:
                doc = json.load(in_f)
                
            text = doc.get("text", "")
            if not text:
                continue

            # Stream chunks from the generator
            chunks = chunk_text(text, chunk_size=300, overlap=50)
            
            for idx, chunk_str in enumerate(chunks):
                base_filename = os.path.basename(file_path).replace(".json", "")
                
                # Propagate metadata so the downstream RAG system can cite its sources
                chunk_data = {
                    "doc_id": base_filename,
                    "chunk_id": f"{base_filename}_chunk_{idx}",
                    "title": doc.get("title", "Unknown Title"),
                    "category": doc.get("category", "Unknown Category"),
                    "source": doc.get("source", "Unknown Source"),
                    "text": chunk_str
                }
                
                # Write individual JSON objects separated by newlines
                out_f.write(json.dumps(chunk_data) + "\n")
                total_chunks_created += 1

            print(f"Processed '{doc.get('title')}' chunks.")
            
    print(f"\nSuccess! Created a total of {total_chunks_created} chunks.")
    print(f"Saved to {output_file}")
    

if __name__ == "__main__":
    process_documents()