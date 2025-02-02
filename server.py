from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Define the data model for input
class Question(BaseModel):
    query: str

# Global variables for model and FAISS index
model = None
sentence_index = None
answers = [
    {"id": "01", "sentence": "Yes I am actively applying and open to work"},
    {"id": "02", "sentence": "You can reach me by phone (404.664.0976) or by email (jawwaad.sabree01@gmail.com)"},
    {"id": "03", "sentence": "I am a developer with over 5 years experience in the industry. I’ve always known I wanted to code and I love building new things"},
    {"id": "04", "sentence": "I am open to freelance, contract, full-time, and part-time positions"},
    {"id": "05", "sentence": "My favorite artist is J. Cole"},
    {"id": "06", "sentence": 'My favorite quote is “sometimes so close can seem so far”'},
    {"id": "07", "sentence": "I love indie-pop, rap, rnb, and old school music"},
    {"id": "08", "sentence": "I prefer dark mode and coding at night"},
    {"id": "09", "sentence": "I am 23 years old"},
    {"id": "10", "sentence": "I began coding at just 12 years old, check my Github if you don’t believe me 😉"},
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, sentence_index  # Ensure global variables persist

    print("🚀 Semantic Search server is starting... Loading model and FAISS index.")

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode answers
    embeddings = model.encode([answer["sentence"] for answer in answers])
    embeddings_vectors = np.array(embeddings).astype("float32")

    # Create FAISS index
    sentence_index = faiss.IndexFlatIP(embeddings_vectors.shape[1])
    sentence_index.add(embeddings_vectors)

    print("✅ Model and FAISS index loaded successfully!")

    yield  # App runs here

    print("🛑 Semantic Search server is shutting down...")

app = FastAPI(lifespan=lifespan)

@app.post("/get_answer/")
async def get_answer(question: Question):
    """Receives a question and returns the closest matching answer."""
    print(f"Received question: {question.query}")
    # Encode query
    query_vector = model.encode([question.query]).astype("float32")
    
    # Search for the most relevant answer
    distances, indices = sentence_index.search(query_vector, 1)
    best_index = indices[0][0]
    confidence = distances[0][0]

    # Normalize confidence
    confidence = round(float((distances[0][0] + 1) / 2) * 100, 2)

    print(f"Best match: {answers[best_index]['sentence']}, Confidence: {confidence}")

    # Return response
    return {
        "question": question.query,
        "bestMatch": answers[best_index]["sentence"],
        "confidence": confidence,
    }
# Run the server with: uvicorn server:app --reload