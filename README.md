# GenAI_NewsResearch_Chatbot
The News Research Tool is an interactive application built with Streamlit, LangChain, and FAISS for efficient and accurate retrieval of information from online news articles. It processes URLs of articles, extracts their content, builds embeddings, and allows users to query the data using an advanced LLM for precise answers and sources.

## Features
- Accepts URLs of news articles via the Streamlit sidebar.
- Loads content from the provided URLs using 'UnstructuredURLLoader'.
-  Splits large text into manageable chunks using 'RecursiveCharacterTextSplitter' for efficient processing.
-  Embeds the document chunks using 'SentenceTransformers' (all-MiniLM-L6-v2) and stores them in a 'FAISS vector store'.
-  Enables users to query the processed data and get accurate responses along with sources using 'ChatGroq LLM'.
