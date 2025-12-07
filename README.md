# What is a Search Engine?

A typo-tolerant, context-aware search engine built on Jeopardy questions and answers spanning 35 seasons. It allows users to search using natural language queries and retrieves the most semantically relevant question-answer pairs — even when queries contain misspellings or are phrased differently from the dataset.

## Features

- **Semantic Search with Sentence Embeddings**  
  Uses `all-MiniLM-L6-v2` from `sentence-transformers` to embed questions and compute similarity.
  
- **Spell Correction with SymSpell**  
  Corrects typos in user queries using a preloaded frequency-based dictionary.
  
- **Cosine Similarity Retrieval**  
  Efficiently retrieves and ranks the top-k most relevant Jeopardy questions based on semantic proximity.

- **Interactive Frontend (HTML + JavaScript)**  
  Clean and modern interface for entering queries and viewing results with relevance scores, categories, and corrected queries.

- **Fully Local and Reproducible**  
  All logic runs offline with open-source models and data. No API keys or remote dependencies.

  ## Collaborators
* [Luis Riviere](https://github.com/LuisAR99)
* [Nikita Sharma](https://github.com/niksisharma)
