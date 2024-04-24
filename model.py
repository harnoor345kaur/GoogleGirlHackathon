import os
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import warnings
warnings.simplefilter("ignore")


class LLMApi(object):
    """
    Creates a class for all the LLM functions
    """
    def load_model(self):
        # Load the token and quantiztion info
        access_token='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'
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it",
                                                       token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
                                                           #quantization_config=quantization_config,
                                                          token=access_token)
        
    def __init__(self):
        # Load the model
        self.load_model()

    
    def _fetch_google_results(self, query):
        # A basic google header
        url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for i, result in enumerate(soup.select('div.tF2Cxc'), 1):
                link = result.select_one('a')['href']
                title = result.select_one('h3').text
                results.append((title, link))
                
                if i == 3:
                    break
                    
            return [query, results]
    
        except requests.RequestException as e:
            print(f"Error fetching results for {query}: {e}")
            return []

    def get_google_results(self, query_list):
        # Placeholder
        all_searches = []

        for query in query_list:
            all_searches.append(self._fetch_google_results(query))

        return all_searches


    def get_suggestions(self, question, user_answer, correct_answer):
        # Define the chat template
        chat = [
            { "role": "user", "content": f"The question was {question}. My answer was {user_answer}. This is not the correct answer and the correct answer was {correct_answer}. Give me the correct answer and the steps required to solve it. Also explain the theory required to answer this question. Give the topic list that i can study to answer similar questions." },
        ]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        # Get the output
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=2000)
        
        # Filter the tokens
        decoded_output = self.tokenizer.decode(outputs[0])
        decoded_output = decoded_output[decoded_output.find("model") + len("model") + 2:]

        # Get the keywords for google search
        list_query = "**Topic list for similar questions:**"
        temp_ = decoded_output.find(list_query)
        keywords = decoded_output[temp_ + len(list_query):-5]
        queries = [i for i in keywords.replace("\n", "").strip().split("*") if len(i) > 2]

        # Return the 
        return decoded_output, queries