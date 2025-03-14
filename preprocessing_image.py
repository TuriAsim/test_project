import pandas as pd
import json
import torch
import numpy as np
import random
import os
import re
import string
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    # Load token from environment variable
    token = os.getenv("HUGGINGFACE_API_KEY")

    if not token:
        raise ValueError("Hugging Face token not found in environment variables!")

    # Log in
    login(token=token)

    df = pd.read_csv('datasets/augmented_tenders.csv')

    # Initialize Ollama LLM
    model_name = "deepseek-ai/DeepSeek-R1"  # Example model identifier
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        use_auth_token=True # Automatically assigns to available GPU if present
    )

    def generate_text(prompt, max_new_tokens=256):
        """
        Generates text output from the model given a prompt.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=256)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    # One-shot prompt template with a demonstration example
    prompt_template = PromptTemplate.from_template(
        """
        You are an expert tender information (Entities) extractor. Follow the example below.

        Example:
        Input Text: '''Status update: Awarded to Suppliers. In a call for competitive bids, Ministry of Education - Schools (Government Ministries & Departments) released Tender No. MOESCHETT18300005 regarding PROVISION OF 
        16-DAY BICULTURAL STUDIES EDUCATIONAL IMMERSION PROGRAMME TO UNITED STATES OF AMERICA FOR DUNMAN HIGH SCHOOL. Following a thorough evaluation, the award was granted on 17/7/2018 to STA TRAVEL (PTE) LTD for $121879.7.
        '''
        
        Output JSON:
        {{
        "Tender NO": "MOESCHETT18300005", 
        "Tender Description": "PROVISION OF 16-DAY BICULTURAL STUDIES EDUCATIONAL IMMERSION PROGRAMME TO UNITED STATES OF AMERICA FOR DUNMAN HIGH SCHOOL",
        "Agency": "Ministry of Education - Schools (Government Ministries & Departments)", 
        "Award Date": "17/7/2018", 
        "Tender Detail Status": "Awarded", 
        "Supplier Name": "STA TRAVEL (PTE) LTD", 
        "Awarded Amount": "$121879.7", 
        "Main Category": "Educational Immersion Programme"
        }}

        Now, extract and classify the following tender information (entities) into structured JSON format.
        Your JSON output must include exactly these entities fields (with these exact keys) and do not give any extra information:
        - Tender NO
        - Tender Description
        - Agency
        - Award Date
        - Tender Detail Status
        - Supplier Name
        - Awarded Amount
        - Main Category

        Text:
        {text}

        JSON output (do not add any extra keys or modify the keys or information):
        """
    )

    def extract_entities(text):
        prompt = prompt_template.format(text=text)
        response = generate_text(prompt, max_new_tokens=768)
        # Clean the response by removing any code block markers if present
        clean_text = re.sub(r"^```json\s*|```$", "", response.strip())
        print("Model output:", clean_text)
        try:
            extracted_json = json.loads(clean_text)
            return extracted_json
        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
            return {"error": clean_text}

    # Example usage:
    # Suppose df is a DataFrame with a column 'augmented_text'
    true_labels = []
    pred_labels = []
    ex_entities = []

    # Uncomment the following loop to process the DataFrame:
    for _, row in df.iterrows():
        unstructured_text = row['augmented_text']
        extracted_entities = extract_entities(unstructured_text)
        ex_entities.append(extracted_entities)
    
    # Save the extracted entities to a new column in the DataFrame        

    # Convert each dictionary to a JSON string using a list comprehension.
    json_strings = [json.dumps(item, ensure_ascii=False) for item in ex_entities]

    # Create a DataFrame with one column called "predicted entities"
    df = pd.DataFrame({"predicted entities": json_strings})

    # Save the DataFrame to a CSV file. Setting index=False avoids writing the DataFrame index to the file.
    df.to_csv("/kaggle/working/deepseek-r1-fewshot_predicted_entities.csv", index=False)
    

if __name__ == "__main__":
    main()


