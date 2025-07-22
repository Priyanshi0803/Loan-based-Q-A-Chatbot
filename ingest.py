import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

def load_csv_to_chunks(csv_path, chunk_cols=None):
    df = pd.read_csv(csv_path)
    chunks = []

    for _, row in df.iterrows():
        if chunk_cols:
            chunk = ' | '.join(str(row[col]) for col in chunk_cols if col in row)
        else:
            chunk = ' | '.join(str(val) for val in row.values)
        chunks.append(chunk)

    return chunks

def create_faiss_index_from_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    with open("faiss_store.pkl", "wb") as f:
        pickle.dump((index, chunks, model), f)

    print("FAISS index created with CSV rows.")

if __name__ == "__main__":
    csv_path = "C:/Users/asinf/OneDrive/Desktop/Assignment-8/Training dataset.csv"  # Adjust path if needed
    chunks = load_csv_to_chunks(csv_path)
    create_faiss_index_from_chunks(chunks)
