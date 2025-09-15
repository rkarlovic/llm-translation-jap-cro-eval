import json
import pandas as pd
from litellm import completion
# import re

japanese_corpus = pd.read_csv("corpus.csv")

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
    except Exception as e:
        print(f"Error making API call: {e}")
        return None