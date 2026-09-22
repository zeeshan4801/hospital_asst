import streamlit as st
import os
import subprocess
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ==========================
# Page Settings
# ==========================

st.set_page_config(
    page_title="Hospital AI Assistant",
    page_icon="🏥"
)

st.title("🏥 Hospital Knowledge Assistant")


# ==========================
# Build FAISS if missing
# ==========================

FAISS_PATH = "faiss_index/index.faiss"
CHUNKS_PATH = "faiss_index/chunks.pkl"


def create_database():

    st.info("Creating knowledge base from PDFs...")

    result = subprocess.run(
        ["python", "ingest.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        st.error(
            "Ingestion failed:"
        )

        st.code(
            result.stderr
        )

        st.stop()



if not os.path.exists(FAISS_PATH):

    create_database()



# After creation check again

if not os.path.exists(FAISS_PATH):

    st.error(
        """
FAISS index was not created.

Check:
1. hospital_knowledge_base folder exists
2. PDF files are uploaded
3. ingest.py works
"""
    )

    st.stop()



# ==========================
# Load Database
# ==========================

@st.cache_resource
def load_database():

    index = faiss.read_index(
        FAISS_PATH
    )


    with open(
        CHUNKS_PATH,
        "rb"
    ) as f:

        chunks = pickle.load(f)



    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    return index, chunks, model



index, chunks, model = load_database()



# ==========================
# Groq Client
# ==========================

client = OpenAI(

    api_key=st.secrets["GROQ_API_KEY"],

    base_url="https://api.groq.com/openai/v1"

)



# ==========================
# Search
# ==========================

def search_documents(question):


    embedding = model.encode(
        [question],
        normalize_embeddings=True
    )


    embedding = np.array(
        embedding
    ).astype("float32")



    scores, ids = index.search(
        embedding,
        5
    )


    results=[]


    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        results.append(
            chunks[idx]
        )


    return results



# ==========================
# Generate Answer
# ==========================

def ask(question):


    docs = search_documents(question)


    context=""


    for d in docs:

        context += f"""

SOURCE:
{d['source']}

PAGE:
{d['page']}

CONTENT:
{d['text']}

-------------------

"""


    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        temperature=0.1,

        messages=[

            {
                "role":"system",
                "content":
                "Answer only from hospital documents."
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



# ==========================
# Chat UI
# ==========================

question = st.chat_input(
    "Ask hospital policy question..."
)


if question:


    answer, sources = ask(question)


    st.subheader("Answer")

    st.write(answer)


    st.subheader("📚 Sources")


    for s in sources:

        st.write(
            f"📄 {s['source']} | Page {s['page']}"
        )
