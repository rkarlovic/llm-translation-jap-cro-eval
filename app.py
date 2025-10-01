import json
import pandas as pd
from litellm import completion
# import re

japanese_corpus = pd.read_csv("corpus.csv")

model_names = ["ollama/mistral:7b"]
# print(japanese_corpus)

def get_response(usr_messages, model_name):
    system_message = {"role": "user", 
                      "content": "This is system prompt"}
    
    if not usr_messages or usr_messages[0]["role"] != "system":
        usr_messages.insert(0, system_message)

    try:
        response = completion(
            model=model_name,
            messages = usr_messages,
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
    
for model_name in model_names:
    results = []
    print(f"Evaluating model: {model_name}")

    for index, row in japanese_corpus.iterrows():
        print(f"Processing row {index + 1}/{len(japanese_corpus)}")
        user_input = [
            {"role": "user", "content": row["input"]}
        ]

        response = get_response(user_input, model_name)

        try:
            response_json = json.loads(response)
            print(f"Response JSON for row {index + 1}: {response_json}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error for row {index + 1}: {e}")
            response_json = {"output": "", "score": 0, "comment": "Invalid JSON response"}
