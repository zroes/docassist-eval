import os
import glob
import json

def chunk_text(text, chunk_size=300, overlap=50):
    """
    Takes a single long string of text and splits it into smaller chunks.
    
    TODO 1: Split the text into a list of words.
    TODO 2: Loop through the list of words, grabbing `chunk_size` words at a time.
    TODO 3: Make sure each chunk overlaps with the previous one by `overlap` words.
    TODO 4: Join the words back into strings and return them as a list.
    """
    # Basically we want a list of strings that each contain 300 words,
    # with a 50 word overlap between each string.
    # How do we iterate over a string and grab a specifed number of words?
    words = text.split() # splits over whitespace
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        yield " ".join(chunk)


def process_documents():
    """
    TODO 1: Get a list of all your .json files in the data/raw/ folder
    
    TODO 2: Open data/processed/chunks.jsonl in "write" mode
    
    TODO 3: Loop through your raw files:
         a. Open the file and load the JSON data
         b. Extract the 'text' field
         c. Pass the text into your `chunk_text` generator
         d. For every chunk yielded, create a new dictionary with:
             # - doc_id (maybe the original filename)
             # - chunk_id (doc_id + an index number)
             # - title, category, source (copied from the original doc)
             # - text (the chunk string)
         e. Convert that new dictionary to a JSON string and write it to the .jsonl file
    """
    raw_json_files = glob.glob("../data/raw/*.json")
    output_file = "../data/processed/chunks.jsonl"

    total_chunks_created = 0

    print(f"{len(raw_json_files)} json files found")
    with open(output_file, "w", encoding="utf-8") as out_f:

        for file_path in raw_json_files:
            with open(file_path, "r", encoding="utf-8") as in_f:

                doc = json.load(in_f)
            text = doc.get("text", "")
            if not text:
                continue
            chunks = chunk_text(text, chunk_size=300, overlap=50)


            for idx, chunk_str in enumerate(chunks):
                
                # We create a unique chunk ID (e.g., "doc_01_chunk_0")
                base_filename = os.path.basename(file_path).replace(".json", "")
                chunk_id = f"{base_filename}_chunk_{idx}"
                
                chunk_data = {
                    "doc_id": base_filename,
                    "chunk_id": chunk_id,
                    "title": doc.get("title", "Unknown Title"),
                    "category": doc.get("category", "Unknown Category"),
                    "source": doc.get("source", "Unknown Source"),
                    "text": chunk_str
                }
                
                # 4. Save to JSONL (JSON Lines format)
                # JSONL means one complete JSON object per line, separated by a newline
                out_f.write(json.dumps(chunk_data) + "\n")
                total_chunks_created += 1

            print(f"Processed '{doc.get('title')}' chunks.")
            
    print(f"\nSuccess! Created a total of {total_chunks_created} chunks.")
    print(f"Saved to {output_file}")
    

if __name__ == "__main__":
    process_documents()