import numpy as np

def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array of shape (50,).
    """
    embeddings = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Split the line into word and vector components
                parts = line.split()
                if not parts:
                    continue
                word = parts[0]
                # Convert the rest of the values into a numpy float array
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[word] = vector
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None
    return embeddings


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors.

    Returns a float in [-1, 1]. If either vector has zero norm, return 0.0.
    """
    # Compute the Euclidean norm (magnitude) of the vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Edge case: Return 0.0 if either vector is a zero vector to avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Formula: (A dot B) / (||A|| * ||B||)
    return np.dot(vec1, vec2) / (norm1 * norm2)


def nearest_neighbors(word, embeddings, n=5):
    """Find the n most similar words to the given word.

    Returns a list of (word, score) tuples sorted by similarity descending,
    excluding the query word itself.
    """
    if word not in embeddings:
        return []

    query_vec = embeddings[word]
    similarities = []

    for other_word, other_vec in embeddings.items():
        # Exclude the query word itself
        if other_word == word:
            continue
        
        # Calculate similarity score
        score = cosine_similarity(query_vec, other_vec)
        similarities.append((other_word, score))

    # Sort results by similarity score (index 1 of the tuple) in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the top n results
    return similarities[:n]


if __name__ == "__main__":
    # Ensure the path matches your repo structure (data/glove_50k_50d.txt)
    glove = load_glove("data/glove_50k_50d.txt")
    
    if glove:
        print(f"Loaded {len(glove)} word vectors")

        # Task 2: Word similarity
        sim = cosine_similarity(glove.get("king", np.zeros(50)),
                                glove.get("queen", np.zeros(50)))
        if sim is not None:
            print(f"cosine('king', 'queen') = {sim:.4f}")

        sim2 = cosine_similarity(glove.get("king", np.zeros(50)),
                                 glove.get("banana", np.zeros(50)))
        if sim2 is not None:
            print(f"cosine('king', 'banana') = {sim2:.4f}")

        # Task 3: Nearest neighbors
        neighbors = nearest_neighbors("king", glove, n=5)
        if neighbors:
            print(f"Nearest to 'king': {neighbors}")