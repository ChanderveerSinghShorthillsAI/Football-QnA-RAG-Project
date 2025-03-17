import faiss
import json
import numpy as np

VECTOR_DB_PATH = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/faiss_index"
OUTPUT_JSON = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/faiss_vector.json"

# Load FAISS index
index = faiss.read_index(VECTOR_DB_PATH)

# Convert FAISS index to NumPy array
num_vectors = index.ntotal
vectors = np.array([index.reconstruct(i) for i in range(num_vectors)])

# Save as JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(vectors.tolist(), f, indent=4)

print(f"✅ FAISS vectors saved to {OUTPUT_JSON}")
