# Retrieval-and-recommendation-ml-algorithms
Python experiments in search ranking (TF-IDF, BM25) and recommender systems (matrix factorization, collaborative filtering), implemented from scratch.


# Retrieval and Recommendation ML Algorithms

Python implementations of classic retrieval and recommendation algorithms,
built with NumPy, SciPy and NetworkX.

## What's inside

### 1. Text search engine (1,000 song lyrics)
- Tokenizer: lowercasing, removing section tags, stripping punctuation
- Inverted index with Boolean retrieval (AND / OR / NOT)
- Positional index for phrase queries
- Ranked retrieval: TF-IDF sum, TF-IDF cosine similarity, BM25
- Side-by-side comparison of how each method ranks the same queries

### 2. Link analysis on an artist graph
- ~7K artists as nodes, edges between artists who share a genre tag
- PageRank, personalized PageRank (seeded on one artist) and HITS (via NetworkX)

### 3. Recommender systems (MovieLens 1M, 70/30 random split)
- Non-personalized movie-average model
- Baseline estimate (global mean + user bias + item bias)
- User-user collaborative filtering (Pearson, 5 neighbors)
- Matrix factorization trained with SGD and L2 regularization
- Matrix factorization with learned user/item biases

| Model | Test RMSE |
|---|---|
| Movie average | 0.979 |
| Baseline estimate | 0.936 |
| User-user CF (k=5) | 1.066 |
| Matrix factorization | ~0.900 |
| MF + biases | ~0.873 |

Numbers vary slightly between runs because the split is random.

## Key takeaways
- Adding user bias is a cheap, large gain over the plain item average.
- User-user CF with only 5 neighbors did worse than the baseline: sparse
  data and noisy similarities.
- Learned biases on top of MF gave the best result.
- BM25 and cosine handle document length differently from a raw TF-IDF sum,
  which changes the top results for multi-word queries.

## Running it
```
pip install numpy scipy pandas networkx matplotlib
```
Datasets are not included: get MovieLens 1M from GroupLens, and the Genius
lyrics data from the original source.

## Notes
Course project work (Texas A&M,). PageRank and HITS use NetworkX;
the search and recommender code is my own implementation.
