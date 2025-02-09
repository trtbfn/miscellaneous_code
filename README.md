# Miscellaneous Code Repository

This repository contains various pieces of code that can be useful for a range of tasks. The focus here is on natural language processing (NLP) utilities, but the collection may expand over time.

## Repository Structure

### NLP Related

This section includes code that is specifically designed for NLP applications.

#### `polars_word2vec_batches.py`
- **Description:**  
  A template script for creating batches suitable for word2vec and pointwise mutual information (PMI) calculations using [Polars](https://www.pola.rs/). This script facilitates the preparation of data batches that can be directly fed into NLP models.

#### `embeddigs_builder.py`
- **Description:**  
  This script builds an index that allows you to measure similarity between documents using high-level document embeddings inspired by min-hash techniques.
- **Features:**
  - Supports internal language models (TF-IDF and BM25) to compute a sketch-based IDF (SIDF) matrix.
  - Constructs two different IDF matrices based on:
    - Count-Min Sketch (CMS)
    - Count Sketch (CS)

