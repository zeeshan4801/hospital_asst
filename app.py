import streamlit as st
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ==========================
# Page Setup
# ==========================

st.set_page_config(
    page_title="Hospital AI Assistant",
    page_icon="🏥"
)


st.title("🏥 Hospital Knowledge Assistant")

st.write(
    "Ask questions from hospital policy documents."
)


# ==========================
# Groq Client
# ==========================

client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)



# ==========================
# Load Database
# ==========================

@st.cache_resource
def load_database():

    index_path = "faiss_index/index.faiss"
    chunks_path = "faiss_index/chunks.pkl"


    # Load FAISS

    index = faiss.read_index(
        index_path
    )


    # Load chunks

    with open(
        chunks_path,
        "rb"
    ) as file:

        chunks = pickle.load(file)



    # Load embedding model

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    return index, chunks, model



index, chunks, model = load_database()



# ==========================
# Search Documents
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

        doc = chunks[idx]


        results.append({

            "text": doc.page_content,

            "source":
            doc.metadata.get(
                "source",
                "Unknown"
            ),

            "page":
            doc.metadata.get(
                "page",
                "Unknown"
            ),

            "score":
            float(score)

        })


    return results



# ==========================
# Generate Answer
# ==========================

def generate_answer(question):


    docs = search_documents(
        question
    )


    context = ""


    for doc in docs:

        context += f"""

SOURCE:
{doc['source']}

PAGE:
{doc['page']}

CONTENT:
{doc['text']}

------------------

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
Do not invent information.
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



# ==========================
# Chat UI
# ==========================


question = st.chat_input(
    "Ask a hospital policy question..."
)



if question:


    with st.chat_message("user"):

        st.write(question)



    with st.chat_message("assistant"):


        with st.spinner(
            "Searching policies..."
        ):


            answer, sources = generate_answer(
                question
            )


        st.write(answer)


        st.divider()

        st.subheader("📚 Sources")


        shown=set()


        for s in sources:


            name = (
                s["source"],
                s["page"]
            )


            if name not in shown:

                st.write(
                    f"""
📄 {s['source']}  
Page: {s['page']}
"""
                )

                shown.add(name)
