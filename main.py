import os
import streamlit as st
import time
import pickle
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

st.title("Finance Info Tool")
st.sidebar.title("Urls")

urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

file_path="db_index.pkl"
main_placeholder=st.empty()

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

if process_url_clicked:
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data=loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    splits=text_splitter.split_documents(data)

    hf_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(splits, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    with open(file_path,"wb") as f:
        pickle.dump(db,f)

    query=main_placeholder.text_input("Question: ")
    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                db_vectorIndex = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=db_vectorIndex.as_retriever())
                answer=chain({'Question':query},return_only_outputs=True)
                st.header("Answer")
                st.write(result["answer"])
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n") 
                    for source in sources_list:
                        st.write(source)



    

