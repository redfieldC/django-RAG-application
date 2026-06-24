import os
from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.chains import (
    RetrievalQA, 
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.retrievers.multi_query import MultiQueryRetriever
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)



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
    for doc in documents:
        doc.metadata["filename"] = os.path.basename(file_path)
        doc.metadata["collection_name"] = collection_name

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


def query_document(question: str, collection_name: str, filename: str = None, chat_history=None) -> dict:
    """
    Query against stored document chunks using RAG.
    Returns LLM answer based on retrieved context.
    """
    if chat_history is None:
        chat_history = []
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


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Given a chat history and the latest user question,
                formulate a standalone question which can be understood
                without the chat history.

                Do NOT answer the question.

                Only rewrite it if needed.
                """
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant.

                Answer ONLY using the information present
                in the provided context.

                Do not use external knowledge.

                If the answer cannot be found in the context,
                respond with:

                "I could not find that information in the document."

                Context:
                {context}
                """
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )


    search_kwargs = {"k": 4}

    if filename:
        search_kwargs["filter"] = {
            "filename": filename
        }

    # Step 4: Build RAG chain
    # retriever fetches top 4 most relevant chunks for the question

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            **search_kwargs,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm,
        multi_query_retriever,
        contextualize_q_prompt
    )

    rewriter_chain = contextualize_q_prompt | llm

    rewritten_question = rewriter_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history
        }
    )

    print("\n" + "=" * 50)
    print("REWRITTEN QUESTION:")
    print(rewritten_question.content)
    print("=" * 50 + "\n")

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    result = rag_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history
        }
    )

    print("=" * 50)
    print(result)
    print("=" * 50)
    sources = []

    for doc in result["context"]:
        sources.append({
            "filename": doc.metadata.get("filename", "Unknown"),
            "page": doc.metadata.get("page", "Unknown"),
            "content": doc.page_content[:200]
        })

    pages = sorted(
        set(
            doc.metadata.get("page")
            for doc in result["context"]
        )
    )
    return {
        "answer": result["answer"],
        "pages": pages,
        "sources": sources
    }