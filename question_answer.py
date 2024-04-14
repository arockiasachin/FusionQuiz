# question_answer.py
import json
import openai

# Load OpenAI API key
from config import OPENAI_API_KEY

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def query_openai(question, documents):
    # Define the prompt including the question and documents
    prompt = f"Question: {question}\nContext: {documents}\nAnswer:"

    # Query OpenAI's API for the answer
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Question Answering AI Assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract the summary from the response
    summary = response.choices[0].message.content
    
    return summary

def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
