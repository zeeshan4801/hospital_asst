import os
import pickle
import numpy as np

import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer



PDF_FOLDER = "hospital_knowledge_base"

OUTPUT_FOLDER = "faiss_index"



def read_pdfs():

    documents = []


    for filename in os.listdir(PDF_FOLDER):

        if filename.lower().endswith(".pdf"):


            path = os.path.join(
                PDF_FOLDER,
                filename
            )


            print("Reading:", filename)


            reader = PdfReader(path)


            for page_number, page in enumerate(reader.pages):

                text = page.extract_text()


                if text:

                    documents.append({

                        "text": text,

                        "source": filename,

                        "page": page_number + 1

                    })


    return documents



def create_chunks(documents):

    chunks = []

    chunk_size = 800


    for doc in documents:


        text = doc["text"]


        for i in range(
            0,
            len(text),
            chunk_size
        ):


            chunks.append({

                "text":
                text[i:i+chunk_size],

                "source":
                doc["source"],

                "page":
                doc["page"]

            })


    return chunks



def create_faiss(chunks):


    print("Creating embeddings...")


    model = SentenceTransformer(

        "sentence-transformers/all-MiniLM-L6-v2"

    )


    texts = [

        c["text"]

        for c in chunks

    ]


    embeddings = model.encode(

        texts,

        normalize_embeddings=True

    )


    embeddings = np.array(

        embeddings

    ).astype("float32")



    index = faiss.IndexFlatIP(

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

    print(
        "Total chunks:",
        len(chunks)
    )




if __name__ == "__main__":


    docs = read_pdfs()


    print(
        "Pages:",
        len(docs)
    )


    chunks = create_chunks(docs)


    print(
        "Chunks:",
        len(chunks)
    )


    create_faiss(chunks)
