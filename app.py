import streamlit as st
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI



# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Hospital AI Assistant",
    page_icon="🏥"
)


st.title("🏥 Hospital Knowledge Assistant")

st.caption(
    "AI assistant powered by hospital policy documents"
)



# =========================
# Groq Client
# =========================

if "GROQ_API_KEY" not in st.secrets:

    st.error(
        "GROQ_API_KEY missing. Add it in Streamlit Secrets."
    )

    st.stop()



client = OpenAI(

    api_key=st.secrets["GROQ_API_KEY"],

    base_url="https://api.groq.com/openai/v1"

)



# =========================
# Load Database
# =========================

@st.cache_resource
def load_database():


    index_file = "faiss_index/index.faiss"

    chunks_file = "faiss_index/chunks.pkl"



    if not st.session_state.get("checked", False):

        st.session_state.checked = True



    try:

        index = faiss.read_index(
            index_file
        )


    except Exception as e:

        st.error(
            f"FAISS index loading failed: {e}"
        )

        st.stop()



    try:

        with open(
            chunks_file,
            "rb"
        ) as f:

            chunks = pickle.load(f)


    except Exception as e:

        st.error(
            f"chunks.pkl loading failed: {e}"
        )

        st.stop()



    model = SentenceTransformer(

        "sentence-transformers/all-MiniLM-L6-v2"

    )


    return index, chunks, model




index, chunks, embedding_model = load_database()



# =========================
# Retrieve Documents
# =========================

def retrieve(question, top_k=5):


    vector = embedding_model.encode(

        [question],

        normalize_embeddings=True

    )


    vector = np.array(
        vector
    ).astype("float32")



    scores, ids = index.search(

        vector,

        top_k

    )



    results=[]



    for score, idx in zip(
        scores[0],
        ids[0]
    ):


        doc = chunks[idx]


        results.append({

            "text":
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



# =========================
# Generate Answer
# =========================

def generate_answer(question):


    docs = retrieve(question)



    context = ""



    for doc in docs:


        context += f"""

SOURCE:
{doc['source']}

PAGE:
{doc['page']}


CONTENT:

{doc['text']}

------------------------

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

If the answer is unavailable,
say:
'I could not find this information in the hospital knowledge base.'
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



# =========================
# Chat Interface
# =========================

if "messages" not in st.session_state:

    st.session_state.messages=[]



for message in st.session_state.messages:


    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )



question = st.chat_input(
    "Ask hospital policy question..."
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
            "Searching documents..."
        ):


            answer, sources = generate_answer(
                question
            )



        st.write(answer)



        st.divider()


        st.subheader(
            "📚 Source Documents"
        )


        shown=set()



        for source in sources:


            key=(

                source["source"],

                source["page"]

            )


            if key not in shown:


                st.write(

                    f"""
📄 **{source['source']}**

Page: {source['page']}

Similarity:
{source['score']:.3f}
"""

                )


                shown.add(key)



    st.session_state.messages.append({

        "role":"assistant",

        "content":answer

    })
