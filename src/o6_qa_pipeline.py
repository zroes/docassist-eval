import csv
from o4_retrieve import get_chroma_collection, retrieve_chunks, format_retrieval_results
from o5_generate import generate_answer

def save_to_csv(data, filepath="../results/prompt_comparison.csv"):
    """
    Args:
    data (list): A list of dictionaries
    filepath (string): the path to save the csv data
    """
    with open(filepath, "w") as csvfile:
        fieldnames = ["question", "prompt_version", "retrieved_docs", "answer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def run_benchmark():

    benchmark_questions =[
        # --- Conceptual Domain Questions ---
        "What are the main components and steps to implement a Retrieval-Augmented Generation (RAG) pattern solution?",
        "What are the key stages in the implementation workflow for a generative AI solution?",
        "What is the difference between working with tools versus writing custom code for foundation models?",
        
        # --- Technical / Code-focused Questions ---
        "How do I interact with foundation models in a conversational format using Python?",
        "Is it possible to stream the generated text output instead of waiting for a single response? How do I do it?",
        "How can I create text embeddings programmatically using the API?",
        "How can I retrieve a list of supported foundation models and find their model IDs?",
        
        # --- Out-of-Domain Questions (To test your 'I don't know' guardrails!) ---
        "How do I bake a classic chocolate chip cookie?",
        "What is the capital of France?",
        "Who won the Super Bowl in 2024?"
    ]


    collection = get_chroma_collection()
    results_list = []
    chunks = retrieve_chunks(collection=collection, queries=benchmark_questions)
    formatted_chunks = format_retrieval_results(queries=benchmark_questions, retrieved_data=chunks)
    
    for i, question in enumerate(benchmark_questions):
        print(f"Processing question {i+1}/10...")
        
        chunks_data = formatted_chunks[i].get("top_chunks",[])
        
        # 1. Extract the titles of the retrieved chunks so we can save them in the CSV
        # We use a quick list comprehension, then join them into a single string.
        titles = [chunk.get('title') for chunk in chunks_data]
        titles_string = ", ".join(titles) if titles else "No docs retrieved (Filtered by distance)"
        
        # 2. Generate the PLAIN answer and append as a dictionary
        plain_answer = generate_answer(
            question=question, 
            retrieved_chunks=chunks_data, 
            prompt_version="plain"
        )
        
        results_list.append({
            "question": question,
            "prompt_version": "plain",
            "retrieved_docs": titles_string,
            "answer": plain_answer
        })
        
        # 3. Generate the GROUNDED answer and append as a dictionary
        grounded_answer = generate_answer(
            question=question, 
            retrieved_chunks=chunks_data, 
            prompt_version="grounded"
        )
        
        results_list.append({
            "question": question,
            "prompt_version": "grounded",
            "retrieved_docs": titles_string,
            "answer": grounded_answer
        })

    # Save the properly formatted dictionaries to CSV!
    save_to_csv(results_list)
    print("\n✅ Benchmark complete! Check results/prompt_comparison.csv")


if __name__ == "__main__":
    run_benchmark()