import streamlit as st
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI



# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Hospital Knowledge Assistant",
    page_icon="🏥",
    layout="centered"
)


# ==============================
# Groq Client
# ==============================

client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)



# ==============================
# Load FAISS Database
# ==============================

@st.cache_resource
def load_database():

    index = faiss.read_index(
        "faiss_index/index.faiss"
    )


    with open(
        "faiss_index/chunks.pkl",
        "rb"
    ) as f:

        chunks = pickle.load(f)


    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    return index, chunks, embedding_model



index, chunks, embedding_model = load_database()



# ==============================
# Retrieve Relevant Chunks
# ==============================

def retrieve_documents(question, top_k=5):


    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )


    scores, ids = index.search(
        np.array(query_embedding),
        top_k
    )


    results = []


    for score, idx in zip(scores[0], ids[0]):

        doc = chunks[idx]


        results.append({

            "content":
                doc.page_content,

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



# ==============================
# Generate Answer
# ==============================

def generate_answer(question):


    documents = retrieve_documents(question)


    context = ""


    for doc in documents:

        context += f"""

SOURCE:
{doc['source']}

PAGE:
{doc['page']}

CONTENT:
{doc['content']}

----------------------

"""



    prompt = f"""

You are a hospital policy assistant.

Answer the user question ONLY using the provided context.

Rules:

- Do not invent information.
- Do not use outside knowledge.
- If the answer is not found, say:
"I could not find this information in the hospital knowledge base."

Provide a clear professional answer.


CONTEXT:

{context}


QUESTION:

{question}

"""


    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

            {
                "role":"system",
                "content":
                "You are an accurate hospital knowledge assistant."
            },

            {
                "role":"user",
                "content":prompt
            }

        ],

        temperature=0.1

    )


    return (
        response.choices[0].message.content,
        documents
    )



# ==============================
# UI
# ==============================

st.title("🏥 Hospital Knowledge Assistant")

st.caption(
    "Ask questions from hospital policies and standards"
)



if "messages" not in st.session_state:

    st.session_state.messages = []



# Display chat history

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )



# User Input

question = st.chat_input(
    "Ask a hospital policy question..."
)



if question:


    st.session_state.messages.append({

        "role":"user",

        "content":question

    })


    with st.chat_message("user"):

        st.write(question)



    with st.chat_message("assistant"):


        with st.spinner(
            "Searching hospital knowledge base..."
        ):


            answer, sources = generate_answer(
                question
            )


        st.write(answer)



        st.divider()


        st.subheader("📚 Sources")


        unique_sources = set()


        for source in sources:


            source_name = (
                source["source"],
                source["page"]
            )


            if source_name not in unique_sources:

                st.write(
                    f"📄 **{source['source']}**  \n"
                    f"Page: {source['page']}  \n"
                    f"Similarity: {source['score']:.3f}"
                )


                unique_sources.add(
                    source_name
                )


    st.session_state.messages.append({

        "role":"assistant",

        "content":answer

    })
