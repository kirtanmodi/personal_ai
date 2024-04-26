import openai
import numpy as np
import pandas as pd
import os


with open('data.txt', 'r') as file:
    about_me_text = file.read()


api_key = os.getenv("OPENAI_API_KEY")


client = openai.OpenAI(api_key=api_key)


def create_embeddings(text):

    response = client.embeddings.create(
        input=text, model="text-embedding-ada-002")

    embedding = response.data[0].embedding
    return embedding


about_me_embeddings = create_embeddings(about_me_text)


def ask_question(question, some_threshold=0.5):
    question_embedding = create_embeddings(question)

    cosine_similarity = np.dot(about_me_embeddings, question_embedding) / (
        np.linalg.norm(about_me_embeddings) * np.linalg.norm(question_embedding))

    # print(f"Cosine Similarity: {cosine_similarity}")

    # if cosine_similarity > some_threshold:
    #     print("Question is closely related to the content.")
    # else:
    #     print("Question may not be very relevant to the content.")

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Given the text: '{
            about_me_text}'\n\nQuestion: {question}\nAnswer:",
        max_tokens=150
    )

    return response.choices[0].text.strip()


while True:
    user_input = input("How can I help you? (Press 'q' to quit):")
    if user_input.lower() == 'q':
        print("Exiting the conversation.")
        break
    else:
        print(ask_question(user_input))
