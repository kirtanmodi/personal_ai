import openai
import numpy as np
import pandas as pd
import os

# Load the text file
with open('data.txt', 'r') as file:
    about_me_text = file.read()

# Fetch the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with the API key
client = openai.OpenAI(api_key=api_key)

# Function to create embeddings


def create_embeddings(text):
    # Call to create embeddings
    response = client.embeddings.create(
        input=text, model="text-embedding-ada-002")
    # Accessing the embedding directly from the response
    embedding = response.data[0].embedding
    return embedding


# Generate embeddings for your text
about_me_embeddings = create_embeddings(about_me_text)

# Function to ask a question


def ask_question(question, some_threshold=0.5):  # Example threshold value
    question_embedding = create_embeddings(question)
    # Calculate cosine similarity
    cosine_similarity = np.dot(about_me_embeddings, question_embedding) / (
        np.linalg.norm(about_me_embeddings) * np.linalg.norm(question_embedding))

    if cosine_similarity > some_threshold:
        print("Question is closely related to the content.")
    else:
        print("Question may not be very relevant to the content.")

    # Getting an answer using GPT model
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Given the text: '{
            about_me_text}'\n\nQuestion: {question}\nAnswer:",
        max_tokens=150
    )

    return response.choices[0].text.strip()


# Example usage
print(ask_question("What are my main interests?"))
