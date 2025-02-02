from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

# Define the data model for input
class Question(BaseModel):
    query: str

# Global variables for model and FAISS index
model = None
sentence_index = None
answers = [
    {"id": "01", "sentence": "Yes I am actively applying and open to work"},
    {"id": "02", "sentence": "You can reach me by phone or email"},
    {"id": "03", "sentence": "I am a developer with over 5 years experience in the industry."},
    {"id": "04", "sentence": "I am open to freelance, contract, full-time, and part-time positions"},
    {"id": "05", "sentence": "My favorite artist is J. Cole"},
    {"id": "06", "sentence": "My favorite quote is 'sometimes so close can seem so far'"},
    {"id": "07", "sentence": "I love indie-pop, rap, rnb, and old school music"},
    {"id": "08", "sentence": "I prefer dark mode and coding at night"},
    {"id": "09", "sentence": "I am 23 years old"},
    {"id": "10", "sentence": "I began coding at just 12 years old, check my GitHub 😉"},
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, sentence_index

    print("🚀 Starting... Loading optimized model and FAISS index.")

    # ✅ Load a smaller model to reduce RAM usage
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")  

    # ✅ Convert embeddings to efficient NumPy array
    embeddings = model.encode([answer["sentence"] for answer in answers], convert_to_numpy=True)
    embeddings_vectors = np.array(embeddings, dtype=np.float32)

    # ✅ Use FAISS IndexHNSWFlat for lower RAM usage
    sentence_index = faiss.IndexHNSWFlat(embeddings_vectors.shape[1], 16)  
    sentence_index.add(embeddings_vectors)

    print("✅ Model & FAISS index loaded successfully!")

    yield  # App runs here

    print("🛑 Server shutting down...")

app = FastAPI(lifespan=lifespan)

# ✅ Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # Allows all headers
)

@app.post("/get_answer/")
async def get_answer(question: Question):
    """Receives a question and returns the closest matching answer."""
    print(f"Received question: {question.query}")

    # Encode query (convert to float32 to match FAISS format)
    query_vector = np.array([model.encode(question.query)], dtype=np.float32)

    # Search for the most relevant answer
    distances, indices = sentence_index.search(query_vector, 1)
    best_index = indices[0][0]
    print(distances[0][0])
    # ✅ Normalize confidence & round to 2 decimal places
    confidence = round(float(100-distances[0][0]), 2)

    print(f"Best match: {answers[best_index]['sentence']}, Confidence: {confidence}%")

    return {
        "question": question.query,
        "bestMatch": answers[best_index]["sentence"],
        "confidence": confidence,
    }

# Run with: uvicorn server:app --workers 2