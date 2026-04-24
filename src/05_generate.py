import os
import google.genai as genai



def build_prompt(question, retrieved_chunks, prompt_version="grounded", distance_threshold=1.5):
    '''
    Args:
        question (string): The user's question
        retrieved_chunks (list): A formatted list of dictionaries with the relevent chunks of information
        prompt_version (string): A flag to decide what system prompt to use
        distance_threshold (float): Used to filter out any irrelevant chunks
    Returns:
        A long string used to prompt an LLM based on the user's question
    '''
    context_list = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        if chunk.get('distance', 10) >= distance_threshold:
            continue
        chunk_block = (
            f"--- Document {i} ---\n"
            f"Title: {chunk.get('title', 'Unknown')}\n"
            f"Source: {chunk.get('source', 'Unknown')}\n"
            f"Content: {chunk.get('text', '')}\n"
        )
        context_list.append(chunk_block)
    
    # Join all the blocks together with a newline
    context_string = "\n".join(context_list)

    if prompt_version == "grounded":
        final_prompt = f"""
        You are a highly reliable, enterprise-grade AI assistant operating in a retrieval-augmented generation (RAG) setting.

        Your task is to answer the user's question using ONLY the information provided in the context below.

        STRICT RULES:
        1. Do NOT use any prior knowledge, assumptions, or external information.
        2. Only rely on the provided documents.
        3. If the answer cannot be found explicitly or inferred directly from the context, respond exactly with:
        "I don't know based on the provided documents."
        4. Do NOT hallucinate, fabricate, or guess.
        5. Be concise and precise. Avoid unnecessary elaboration.
        6. Always cite the document titles used in your answer.
        - Use clear inline citations like: (Title: <document title>)
        - If multiple documents are used, cite each relevant title.

        CONTEXT:
        {context_string}

        USER QUESTION:
        {question}

        ANSWER:
        """


    else:
        final_prompt = f"""
            You are a helpful assistant. Answer the user's question using the provided context.
            Context:
            {context_string}

            Question: {question}
            """
        
    print(final_prompt)
    return final_prompt

def call_llm(prompt, provider="gemini"):
    """
    Model-agnostic function to send a prompt to an LLM and return the text. 
    """
    
    if provider == "gemini":
        # Use export GEMINI_API_KEY="{your gemini api key}"
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No Gemini API key found!")
        
        
        # 2. Configure the client
        client = genai.Client(api_key=api_key)
        
        # 3. Call the model using the new syntax
        # We can use 'gemini-2.5-flash' which is their newest, blazing fast model for text tasks
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text

def generate_answer(question, retrieved_chunks, prompt_version="grounded", provider="gemini"):
    # 1. Build the massive string using your excellent prompt logic
    final_prompt = build_prompt(question, retrieved_chunks, prompt_version=prompt_version)
    
    # 2. Send it to the LLM
    answer = call_llm(final_prompt, provider)
    
    return answer


if __name__ == "__main__":
    mock_question = "How do I chat with a model in Python?"
    mock_chunks =[
        {
            "title": "Quick code tutorial: Chat with a model",
            "source": "https://dataplatform.cloud.ibm.com/docs/chat",
            "text": "You can use the ibm_watsonx_ai library to interact with foundation models. You pass a list of messages with 'role' and 'content'."
        }
    ]
    
    print("🤖 Sending to Gemini...")
    # This will call generate_answer -> build_prompt -> call_llm
    final_answer = generate_answer(mock_question, mock_chunks, prompt_version="grounded", provider="gemini")
    
    print("\n--- FINAL ANSWER ---")
    print(final_answer)