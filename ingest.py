import os
import pickle
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PDF_FOLDER = "hospital_knowledge_base"

OUTPUT_FOLDER = "faiss_index"


def load_pdfs():

    documents = []

    for file in os.listdir(PDF_FOLDER):

        if file.endswith(".pdf"):

            path = os.path.join(
                PDF_FOLDER,
                file
            )

            print("Reading:", file)

            reader = PdfReader(path)


            for page_number, page in enumerate(reader.pages):

                text = page.extract_text()

                if text:

                    documents.append({

                        "text": text,

                        "source": file,

                        "page": page_number + 1

                    })


    return documents



def split_text(documents):

    chunks=[]


    for doc in documents:

        text = doc["text"]


        size = 800


        for i in range(0,len(text),size):

            chunks.append({

                "text":
                text[i:i+size],

                "source":
                doc["source"],

                "page":
                doc["page"]

            })


    return chunks



def create_index(chunks):


    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    texts=[
        c["text"]
        for c in chunks
    ]


    embeddings=model.encode(
        texts,
        normalize_embeddings=True
    )


    embeddings=np.array(
        embeddings
    ).astype("float32")



    index=faiss.IndexFlatIP(
        embeddings.shape[1]
    )


    index.add(
        embeddings
    )


    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )


    faiss.write_index(
        index,
        "faiss_index/index.faiss"
    )


    with open(
        "faiss_index/chunks.pkl",
        "wb"
    ) as f:

        pickle.dump(
            chunks,
            f
        )


    print("FAISS created successfully")



if __name__=="__main__":

    docs=load_pdfs()

    chunks=split_text(docs)

    print(
        "Chunks:",
        len(chunks)
    )

    create_index(chunks)
