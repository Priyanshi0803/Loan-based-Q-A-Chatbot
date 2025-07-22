import faiss
import pickle
import os
from groq import Groq
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Initialize Groq client with correct base_url and secure key handling
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),  # ✅ Don't hardcode your API key
    base_url="https://api.groq.com"     # ✅ FIXED: Removed extra /openai/v1
)

def retrieve(query, k=3):
    with open("faiss_store.pkl", "rb") as f:
        index, chunks, model = pickle.load(f)

    q_emb = model.encode([query])
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

def generate_answer(context, question):
    prompt = f"""Answer the question using the following context:

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",  # ✅ Groq-supported model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()