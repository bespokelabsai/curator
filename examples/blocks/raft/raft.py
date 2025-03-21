import os

import numpy as np
import requests
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from bespokelabs.curator.blocks.raft import Raft
from examples.blocks.raft.utils import extract_text

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 512))
assert CHUNK_SIZE > 0

raft = Raft(model="gpt-4o", distractors=3, n_questions=10, chunk_size=CHUNK_SIZE, p=0.85)

default_pdf_url = "https://arxiv.org/pdf/2503.03323.pdf"
pdf_url = os.environ.get("PDF_URL", default_pdf_url)

try:
    response = requests.get(pdf_url, stream=True)
except requests.exceptions.RequestException as e:
    print(f"Failed to download the pdf: {e}")
    exit(1)


pdf_file = pdf_url.split("/")[-1]
with open(pdf_file, "wb") as f:
    f.write(response.content)

text = extract_text(pdf_file, backend=os.environ.get("OCR_BACKEND", "aryn"))

model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_clusters(questions, model, similarity_threshold=0.85):
    """Computes clusters of similar questions based on embeddings and cosine similarity."""
    print("Generating embeddings for questions...")
    embeddings = model.encode(questions, show_progress_bar=True)

    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)

    print("Clustering similar questions...")
    distances = 1 - similarity_matrix
    distances = np.maximum(distances, 0)

    clustering = DBSCAN(eps=1 - similarity_threshold, min_samples=2, metric="precomputed").fit(distances)
    labels = clustering.labels_

    return embeddings, labels


def remove_redundant_questions(dataset, model, similarity_threshold=0.85):
    """Removes redundant questions from a Hugging Face dataset with QA pairs."""
    questions = dataset["question"]
    embeddings, labels = compute_clusters(questions, model, similarity_threshold)

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

    return dataset.select(filtered_indices)


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings()
    n_of_chunks = len(text) // CHUNK_SIZE
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        number_of_chunks=n_of_chunks,
    )

    chunks = text_splitter.split_text(text)

    dataset = raft(chunks)
    dataset = remove_redundant_questions(dataset, model)

    dataset.to_parquet("raft_dataset.parquet")
    print(dataset[0].keys())
