import os

import torch
from langchain.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.blocks.raft.utils import extract_text

model_path = os.environ.get("MODEL_PATH", "llama3-finetuned/final")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 512))
assert CHUNK_SIZE > 0

default_pdf_url = "https://arxiv.org/pdf/2503.03323.pdf"
pdf_url = os.environ.get("PDF_URL", default_pdf_url)

pdf_file = pdf_url.split("/")[-1]
text = extract_text(pdf_file, backend=os.environ.get("OCR_BACKEND", "aryn"))

embeddings = OpenAIEmbeddings()

n_of_chunks = len(text) // CHUNK_SIZE
text_splitter = SemanticChunker(embeddings=embeddings, number_of_chunks=n_of_chunks)

chunks = text_splitter.split_text(text)

print(f"Created {len(chunks)} chunks")

print("Creating embeddings and vector store...")
vector_store = FAISS.from_texts(chunks, embeddings)


def retrieve_and_generate(query, model, tokenizer, top_k=3):
    """Perform Retrieval-Augmented Generation (RAG) on a PDF file."""
    print(f"Retrieving top {top_k} chunks for query: '{query}'")
    results = vector_store.similarity_search(query, k=top_k)

    context = "".join(["<DOCUMENT>" + doc.page_content + "</DOCUMENT>" for doc in results])
    prompt = f"""Based on the following context from a document, please answer the question.

    Context:
    {context}

    Question: {query}"""
    prompt = f"<|begin_of_text|>[INST] {prompt} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=8192,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        answer = response.split("Answer:")[1].strip()
    except Exception:
        answer = response

    return answer


while True:
    query = input("Enter prompt (Press Ctrl-c to exit): ")
    answer = retrieve_and_generate(
        query,
        model,
        tokenizer,
        top_k=3,
    )

    print("\n" + "=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(answer)
