import os
import subprocess
import pickle

import streamlit as st
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI



st.set_page_config(
    page_title="Hospital AI Assistant",
    page_icon="🏥"
)


st.title("🏥 Hospital Knowledge Assistant")



FAISS_FILE="faiss_index/index.faiss"

CHUNKS_FILE="faiss_index/chunks.pkl"



# Create FAISS automatically

if not os.path.exists(FAISS_FILE):

    with st.spinner(
        "Creating hospital knowledge base..."
    ):

        result=subprocess.run(

            [
                "python",
                "ingest.py"
            ],

            capture_output=True,

            text=True

        )


        if result.returncode != 0:

            st.error(
                result.stderr
            )

            st.stop()



# Load database

@st.cache_resource
def load_database():

    index=faiss.read_index(
        FAISS_FILE
    )


    with open(
        CHUNKS_FILE,
        "rb"
    ) as f:

        chunks=pickle.load(f)


    model=SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    return index,chunks,model



index,chunks,model=load_database()



# Groq

client=OpenAI(

    api_key=st.secrets["GROQ_API_KEY"],

    base_url="https://api.groq.com/openai/v1"

)



def search(question):


    vector=model.encode(

        [question],

        normalize_embeddings=True

    )


    vector=np.array(
        vector
    ).astype("float32")



    scores,ids=index.search(

        vector,

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



def generate_answer(question):


    docs=search(question)


    context=""


    for d in docs:

        context+=f"""

Source:
{d['source']}

Page:
{d['page']}

Content:
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
            f"{context}\nQuestion:{question}"
            }

        ]

    )


    return response.choices[0].message.content, docs



question=st.chat_input(
    "Ask hospital policy question..."
)



if question:


    answer,sources=generate_answer(
        question
    )


    st.write(answer)


    st.subheader(
        "📚 Sources"
    )


    for s in sources:

        st.write(
            f"📄 {s['source']} - Page {s['page']}"
        )
