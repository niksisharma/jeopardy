from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from symspellpy import SymSpell, Verbosity
import os
 
app = Flask(__name__)
 
# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_pickle("data/jeopardy_with_embeddings.pkl")
df['q_embedding'] = df['q_embedding'].apply(lambda x: np.array(x))
 
# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = os.path.join("data", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
 
# Spell correction
def correct_query(query):
    suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
    return suggestions[0].term if suggestions else query
 
# Search function
def search(query, top_k=5):
    query_vec = model.encode(query, convert_to_numpy=True).reshape(1, -1)
    matrix = np.vstack(df['q_embedding'].values)
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices][['question', 'answer', 'category']].copy()
    results['score'] = scores[top_indices]
    return results.to_dict(orient='records')
 
@app.route("/")
def index():
    return render_template("index.html")
 
@app.route("/search", methods=["POST"])
def do_search():
    original_query = request.json.get("query", "")
    if not original_query:
        return jsonify({"results": []})
 
    corrected_query = correct_query(original_query)
    results = search(corrected_query, top_k=5)
    return jsonify({
        "original_query": original_query,
        "corrected_query": corrected_query,
        "results": results
    })
 
if __name__ == "__main__":
    app.run(debug=True)