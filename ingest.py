# ingest.py

import os
import pickle
import numpy as np

# FAISS import
try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is not installed. Install faiss-cpu in requirements.txt"
    )


from pypdf import PdfReader
from sentence_transformers import SentenceTransformer



# ==========================
# Paths
# ==========================

PDF_FOLDER = "hospital_knowledge_base"

OUTPUT_FOLDER = "faiss_index"

INDEX_FILE = os.path.join(
    OUTPUT_FOLDER,
    "index.faiss"
)

CHUNKS_FILE = os.path.join(
    OUTPUT_FOLDER,
    "chunks.pkl"
)



# ==========================
# Read PDFs
# ==========================

def load_pdfs():

    documents = []


    if not os.path.exists(PDF_FOLDER):

        raise FileNotFoundError(
            f"PDF folder not found: {PDF_FOLDER}"
        )


    pdf_files = [

        f for f in os.listdir(PDF_FOLDER)

        if f.lower().endswith(".pdf")

    ]


    if not pdf_files:

        raise FileNotFoundError(
            "No PDF files found in hospital_knowledge_base folder"
        )


    for filename in pdf_files:


        filepath = os.path.join(
            PDF_FOLDER,
            filename
        )


        print(
            f"Reading PDF: {filename}"
        )


        try:

            reader = PdfReader(filepath)


            for page_number, page in enumerate(reader.pages):


                text = page.extract_text()


                if text and text.strip():


                    documents.append({

                        "text": text,

                        "source": filename,

                        "page": page_number + 1

                    })


        except Exception as e:

            print(
                f"Skipping {filename}: {e}"
            )


    print(
        f"Total pages loaded: {len(documents)}"
    )


    return documents



# ==========================
# Split Text
# ==========================

def create_chunks(documents):


    chunks = []


    chunk_size = 800

    overlap = 100



    for doc in documents:


        text = doc["text"]


        start = 0


        while start < len(text):


            chunk_text = text[
                start:start + chunk_size
            ]


            chunks.append({

                "text": chunk_text,

                "source": doc["source"],

                "page": doc["page"]

            })


            start += chunk_size - overlap



    print(
        f"Total chunks created: {len(chunks)}"
    )


    return chunks



# ==========================
# Create FAISS
# ==========================

def create_faiss_index(chunks):


    print(
        "Loading embedding model..."
    )


    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    texts = [

        item["text"]

        for item in chunks

    ]


    print(
        "Creating embeddings..."
    )


    embeddings = model.encode(

        texts,

        batch_size=32,

        show_progress_bar=True,

        normalize_embeddings=True

    )


    embeddings = np.array(
        embeddings
    ).astype("float32")



    print(
        "Embedding shape:",
        embeddings.shape
    )



    # Create FAISS index

    dimension = embeddings.shape[1]


    index = faiss.IndexFlatIP(
        dimension
    )


    index.add(
        embeddings
    )



    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )



    print(
        "Saving FAISS index..."
    )


    faiss.write_index(

        index,

        INDEX_FILE

    )



    with open(
        CHUNKS_FILE,
        "wb"
    ) as f:

        pickle.dump(
            chunks,
            f
        )



    print(
        "FAISS index saved successfully"
    )


    print(
        f"Documents stored: {len(chunks)}"
    )



# ==========================
# Main
# ==========================

if __name__ == "__main__":


    print(
        "Starting ingestion..."
    )


    documents = load_pdfs()


    chunks = create_chunks(
        documents
    )


    create_faiss_index(
        chunks
    )


    print(
        "DONE ✅"
    )
