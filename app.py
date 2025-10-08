import json
import os
import pandas as pd
from litellm import completion
import time
from datetime import timedelta

japanese_corpus = pd.read_csv("corpus.csv")
# "ollama/mistral:7b","ollama/gemma3:latest","ollama/qwen3:latest","ollama/qwen2:latest","ollama/deepseek-r1:latest",
model_names = ["ollama/phi4:latest"]

def get_response(usr_messages, model_name):
    system_message = {"role": "user",
                     "content": """
 You are a professional translator. Your task is to translate text from Japanese into Croatian.
 Pay attention to Croatian grammar, grammatical case and spelling.
 Return strictly only Croatian translation.
 Do not include any other text or explanation.
 """
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
            return response.choices[0].message.content
        else:
            print("No choices found in the response.")
            return None
    except Exception as e:
        print(f"Error making API call: {e}")
        return None

output_dir = "translation_results"
os.makedirs(output_dir, exist_ok=True)

total_start_time = time.time()
model_times = []

for model_name in model_names:
    model_start_time = time.time()
    print(f"\nEvaluating model: {model_name}")
    
    # Prepare output filename
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    output_filename = f"{output_dir}/{safe_model_name}_results.csv"
    
    # Check if file already exists and load existing results
    if os.path.exists(output_filename):
        existing_df = pd.read_csv(output_filename, encoding='utf-8')
        start_index = len(existing_df)
        print(f"Resuming from row {start_index + 1} (found {start_index} existing results)")
    else:
        existing_df = None
        start_index = 0
        # Create file with headers
        headers_df = pd.DataFrame(columns=['input', 'translation', 'llm_results'])
        headers_df.to_csv(output_filename, index=False, encoding='utf-8')
    
    # Process rows starting from where we left off
    for index, row in japanese_corpus.iterrows():
        if index < start_index:
            continue  # Skip already processed rows
            
        print(f"Processing row {index + 1}/{len(japanese_corpus)}")
        
        user_input = [
            {"role": "user", "content": row["input"]}
        ]
        response = get_response(user_input, model_name)
        
        result_row = {
            'input': row['input'],
            'translation': row['translation'],
            'llm_results': response if response else ""
        }
        
        # Append this single result to CSV immediately
        result_df = pd.DataFrame([result_row])
        result_df.to_csv(output_filename, mode='a', header=False, index=False, encoding='utf-8')
        
        print(f"Response text for row {index + 1}: {response}")
        print(f"Saved to {output_filename}")
    
    model_end_time = time.time()
    elapsed_time = model_end_time - model_start_time
    elapsed_timedelta = timedelta(seconds=elapsed_time)
    
    print(f"\nResults for {model_name} saved to {output_filename}")
    print(f"Total rows processed: {len(japanese_corpus)}")
    print(f"Time taken: {elapsed_timedelta} ({elapsed_time:.2f} seconds)")

print("\nAll models evaluated and results saved!")