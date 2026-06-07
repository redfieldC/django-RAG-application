import os
from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pathlib import Path



def get_embeddings():
    """
    Returns HuggingFace embeddings — runs locally, no API key needed, completely free.
    all-MiniLM-L6-v2 is small, fast, and good enough for most RAG use cases.
    """
    
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def ingest_document(file_path: str, collection_name: str) -> int:
    """
    Load PDF, split into chunks, embed and store in ChromaDB.
    Returns number of chunks created.
    """
    
    
    # Step 1: Load the PDF
    file_path = str(Path(file_path).resolve())  

    
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Step 2: Split into chunks
    # chunk_size=1000 means ~1000 characters per chunk
    # chunk_overlap=200 means chunks share 200 characters — helps preserve context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Step 3: Embed and store in ChromaDB
    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )

    return len(chunks)


def query_document(question: str, collection_name: str) -> str:
    """
    Query against stored document chunks using RAG.
    Returns LLM answer based on retrieved context.
    """
    # Step 1: Load existing ChromaDB collection
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )

    # Step 2: Setup Groq LLM
    # llama-3.3-70b-versatile is Groq's fastest, most capable free model
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=settings.GROQ_API_KEY
    )

    # Step 3: Define prompt
    prompt_template = """You are a helpful assistant answering questions based on the provided context.
    Synthesize information from the context to give a complete answer.
    If the context contains relevant information, use it even if the exact phrase from the question isn't present.
    Only say you lack information if the context genuinely contains nothing relevant.

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Step 4: Build RAG chain
    # retriever fetches top 4 most relevant chunks for the question
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = put all chunks into one prompt
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.invoke({"query": question})
    return result["result"]