import streamlit as st
import os
import subprocess
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI



st.set_page_config(

    page_title="Hospital AI Assistant",

    page_icon="🏥"

)


st.title("🏥 Hospital Knowledge Assistant")

st.caption(
    "Ask questions from hospital policy documents"
)



FAISS_FILE = "faiss_index/index.faiss"

CHUNKS_FILE = "faiss_index/chunks.pkl"



# -------------------------
# Create database if missing
# -------------------------

if not os.path.exists(FAISS_FILE):


    with st.spinner(
        "Building hospital knowledge base..."
    ):


        result = subprocess.run(

            [
                "python",
                "ingest.py"
            ],

            capture_output=True,

            text=True

        )



        if result.returncode != 0:


            st.error(
                "Knowledge base creation failed"
            )


            st.code(
                result.stderr
            )


            st.stop()



# -------------------------
# Load database
# -------------------------

@st.cache_resource
def load_database():


    index = faiss.read_index(

        FAISS_FILE

    )


    with open(

        CHUNKS_FILE,

        "rb"

    ) as f:

        chunks = pickle.load(f)



    model = SentenceTransformer(

        "sentence-transformers/all-MiniLM-L6-v2"

    )


    return index, chunks, model




index, chunks, embedding_model = load_database()



# -------------------------
# Groq
# -------------------------

client = OpenAI(

    api_key=st.secrets["GROQ_API_KEY"],

    base_url="https://api.groq.com/openai/v1"

)



def retrieve(question):


    vector = embedding_model.encode(

        [question],

        normalize_embeddings=True

    )


    vector = np.array(

        vector

    ).astype("float32")



    scores, ids = index.search(

        vector,

        5

    )


    results=[]


    for score, idx in zip(

        scores[0],

        ids[0]

    ):


        results.append({

            **chunks[idx],

            "score":float(score)

        })



    return results




def ask(question):


    docs = retrieve(question)



    context = ""



    for d in docs:


        context += f"""

SOURCE:
{d['source']}

PAGE:
{d['page']}

CONTENT:
{d['text']}

-----------------

"""



    response = client.chat.completions.create(


        model="openai/gpt-oss-120b",


        temperature=0.1,


        messages=[


            {

            "role":"system",

            "content":
            """
You are a hospital policy assistant.
Answer only from provided documents.
Do not make assumptions.
"""

            },


            {

            "role":"user",

            "content":
            f"""

Context:

{context}


Question:

{question}

"""

            }

        ]

    )



    return response.choices[0].message.content, docs




question = st.chat_input(

    "Ask your question..."

)



if question:


    with st.chat_message("user"):

        st.write(question)



    with st.chat_message("assistant"):


        with st.spinner(
            "Searching policies..."
        ):


            answer, sources = ask(question)



        st.write(answer)



        st.divider()


        st.subheader("📚 Sources")


        for s in sources:


            st.write(

                f"""
📄 {s['source']}

Page: {s['page']}
"""

            )
