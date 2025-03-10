import os

import numpy as np
import pdfplumber
import requests
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from bespokelabs.curator.blocks.raft import Raft

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 512))
assert CHUNK_SIZE > 0

raft = Raft(model="gpt-4o", distractors=3, n_questions=10, chunk_size=CHUNK_SIZE, p=0.85)
model = SentenceTransformer("all-MiniLM-L6-v2")

arxiv_id = os.environ.get("ARXIV_ID", "2503.03323")  # change this to the arxiv id of the paper you want to test
try:
    response = requests.get(f"https://arxiv.org/pdf/{arxiv_id}.pdf", stream=True)
except requests.exceptions.RequestException as e:
    print(f"Failed to download the paper: {e}")
    exit(1)

with open(f"{arxiv_id}.pdf", "wb") as f:
    f.write(response.content)

with pdfplumber.open(f"{arxiv_id}.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""

with open(f"{arxiv_id}.txt", "w", encoding="utf-8") as f:
    f.write(text)


def remove_redundant_questions(dataset, similarity_threshold=0.85):
    """Removes redundant questions from a Hugging Face dataset with QA pairs."""
    questions = dataset["question"]

    print("Generating embeddings for questions...")
    embeddings = model.encode(questions, show_progress_bar=True)

    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)

    print("Clustering similar questions...")
    distances = 1 - similarity_matrix
    distances = np.maximum(distances, 0)

    clustering = DBSCAN(eps=1 - similarity_threshold, min_samples=2, metric="precomputed").fit(distances)

    labels = clustering.labels_

    questions_to_keep = set()

    for i, label in enumerate(labels):
        if label == -1:
            questions_to_keep.add(i)

    for label in set(labels):
        if label == -1:
            continue

        cluster_indices = [i for i, lab in enumerate(labels) if lab == label]

        cluster_embeddings = embeddings[cluster_indices]
        cluster_center = np.mean(cluster_embeddings, axis=0)

        distances_to_center = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)

        closest_question_idx = cluster_indices[np.argmin(distances_to_center)]
        questions_to_keep.add(closest_question_idx)

    # Filter the dataset
    filtered_indices = sorted(questions_to_keep)

    print(f"Original dataset size: {len(dataset)}")
    print(f"Filtered dataset size: {len(filtered_indices)}")
    print(f"Removed {len(dataset) - len(filtered_indices)} redundant questions")

    # Create a new filtered dataset
    filtered_dataset = dataset.select(filtered_indices)

    return filtered_dataset


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings()
    n_of_chunks = len(text) // CHUNK_SIZE
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        number_of_chunks=n_of_chunks,
    )

    chunks = text_splitter.split_text(text)

    dataset = raft(chunks)
    dataset = remove_redundant_questions(dataset)

    dataset.to_parquet("raft_dataset.parquet")
    print(dataset[0].keys())
