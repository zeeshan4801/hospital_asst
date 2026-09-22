import streamlit as st
import os
import subprocess
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI



st.set_page_config(
    page_title="Hospital AI Assistant",
    page_icon="🏥"
)


st.title("🏥 Hospital Knowledge Assistant")



# -------------------------
# Create FAISS if missing
# -------------------------

if not os.path.exists(
    "faiss_index/index.faiss"
):

    with st.spinner(
        "Creating knowledge base..."
    ):

        subprocess.run(
            ["python","ingest.py"]
        )



# -------------------------
# Load database
# -------------------------

@st.cache_resource
def load_database():


    index=faiss.read_index(
        "faiss_index/index.faiss"
    )


    with open(
        "faiss_index/chunks.pkl",
        "rb"
    ) as f:

        chunks=pickle.load(f)



    model=SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    return index,chunks,model



index,chunks,model=load_database()



# -------------------------
# Groq
# -------------------------

client=OpenAI(

    api_key=st.secrets["GROQ_API_KEY"],

    base_url="https://api.groq.com/openai/v1"

)



def search(question):


    emb=model.encode(
        [question],
        normalize_embeddings=True
    )


    emb=np.array(
        emb
    ).astype("float32")



    scores,ids=index.search(
        emb,
        5
    )


    results=[]


    for score,idx in zip(
        scores[0],
        ids[0]
    ):


        results.append(
            chunks[idx]
        )


    return results



def answer(question):


    docs=search(question)


    context=""


    for d in docs:

        context+=f"""

SOURCE:
{d['source']}

PAGE:
{d['page']}

TEXT:
{d['text']}

"""


    response=client.chat.completions.create(

        model="openai/gpt-oss-120b",

        temperature=0.1,

        messages=[

        {
        "role":"system",
        "content":
        "Answer only from provided hospital documents."
        },

        {
        "role":"user",
        "content":
        f"{context}\n\nQuestion:{question}"
        }

        ]

    )


    return response.choices[0].message.content,docs



question=st.chat_input(
    "Ask your question..."
)


if question:


    reply,sources=answer(question)


    st.write(reply)


    st.subheader("📚 Sources")


    for s in sources:

        st.write(
            f"{s['source']} - Page {s['page']}"
        )
