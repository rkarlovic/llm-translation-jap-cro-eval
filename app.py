import json
import os
import pandas as pd
from litellm import completion
import time
from datetime import timedelta
import re

japanese_corpus = pd.read_csv("corpus.csv")
model_names = ["ollama/mistral:7b","ollama/gemma3:latest","ollama/qwen3:latest","ollama/qwen2:latest", "ollama/deepseek-r1:latest", "ollama/phi4:latest"]

def clean_response(text):
    """Remove thinking blocks and XML-like tags from model responses."""
    if not text:
        return text
    
    # Remove <think>...</think> blocks (case insensitive, handles multiline)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove any other common XML-like tags that might wrap the response
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Strip leading/trailing whitespace and quotes
    text = text.strip().strip('"\'')
    
    return text

def get_response(usr_messages, model_name):
    """Get translation response from the model."""
    system_message = {
        "role": "user",
        "content": """You are a professional translator. Your task is to translate text from Japanese into Croatian.
Pay attention to Croatian grammar, grammatical case and spelling.
Return strictly only Croatian translation.
Do not include any other text or explanation."""
    }
    
    if not usr_messages or usr_messages[0]["role"] != "system":
        usr_messages.insert(0, system_message)
    
    try:
        response = completion(
            model=model_name,
            messages=usr_messages,
            api_base="http://localhost:11434"
        )
        if hasattr(response, 'choices') and response.choices:
            return clean_response(response.choices[0].message.content)
        else:
            print("No choices found in the response.")
            return None
    except Exception as e:
        print(f"Error making API call: {e}")
        return None

# Create output directory
output_dir = "translation_results"
os.makedirs(output_dir, exist_ok=True)

total_start_time = time.time()

for model_name in model_names:
    model_start_time = time.time()
    print(f"\nEvaluating model: {model_name}")
    
    # Prepare output filename
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    output_filename = f"{output_dir}/{safe_model_name}_results.csv"
    
    # Initialize results list
    results = []
    
    # Process all rows
    for index, row in japanese_corpus.iterrows():
        print(f"Processing row {index + 1}/{len(japanese_corpus)}")
        
        user_input = [{"role": "user", "content": row["input"]}]
        response = get_response(user_input, model_name)
        
        results.append({
            'input': row['input'],
            'translation': row['translation'],
            'llm_results': response if response else ""
        })
        
        print(f"Response text for row {index + 1}: {response}")
    
    # Save all results at once (overwrites any existing file)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filename, index=False, encoding='utf-8')
    
    model_end_time = time.time()
    elapsed_time = model_end_time - model_start_time
    elapsed_timedelta = timedelta(seconds=elapsed_time)
    
    print(f"\nResults for {model_name} saved to {output_filename}")
    print(f"Total rows processed: {len(japanese_corpus)}")
    print(f"Time taken: {elapsed_timedelta} ({elapsed_time:.2f} seconds)")

total_end_time = time.time()
total_elapsed = timedelta(seconds=total_end_time - total_start_time)
print(f"\nAll models evaluated and results saved!")
print(f"Total time: {total_elapsed}")