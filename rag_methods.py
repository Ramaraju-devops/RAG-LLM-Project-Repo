import os
import dotenv
from time import time
import streamlit as st

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI  # Azure ONLY
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env file
dotenv.load_dotenv(override=True)

os.environ["USER_AGENT"] = "myagent"

DB_DOCS_LIMIT = 10

# ---- INDEXING PHASE ----

def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})


def load_doc_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        if "rag_sources" not in st.session_state:
            st.session_state.rag_sources = []

        # Initialize chroma client reference for collection pruning
        chroma_client = None
        if "vector_db" in st.session_state:
            chroma_client = st.session_state.vector_db._client

        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                # Limit total documents to 20
                if len(st.session_state.rag_sources) < 20:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())
                    try:
                        if doc_file.type == "application/pdf" or doc_file.name.endswith(".pdf"):
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.name.endswith(".txt") or doc_file.name.endswith(".md"):
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                        # Prune collections if over 20 limit
                        if chroma_client:
                            collection_names = sorted([col.name for col in chroma_client.list_collections()])
                            while len(collection_names) > 20:
                                chroma_client.delete_collection(collection_names[0])
                                collection_names.pop(0)

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")
                    finally:
                        os.remove(file_path)
                else:
                    st.error(f"Maximum number of documents reached (20).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document(s) loaded successfully: {[doc_file.name for doc_file in st.session_state.rag_docs]}", icon="✅")


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if "rag_sources" not in st.session_state:
            st.session_state.rag_sources = []

        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 20:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)
                except Exception as e:
                    st.error(f"Error loading document from URL {url}: {e}")
            else:
                st.error("Maximum number of documents reached (20).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document from URL {url} loaded successfully.", icon="✅")


def initialize_vector_db(docs):
    embedding = AzureOpenAIEmbeddings(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
        openai_api_type="azure",
    )
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state.get('session_id', 'anon'),
    )

    # Clean up collections if >20
    chroma_client = vector_db._client
    collection_names = sorted([col.name for col in chroma_client.list_collections()])
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db


def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    document_chunks = text_splitter.split_documents(docs)
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(document_chunks)
    else:
        st.session_state.vector_db.add_documents(document_chunks)


# ---- RAG PHASE ----

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. You will have to answer users' queries. You will have some context to help with your answers, but not always will it be completely related or helpful. You can also use your knowledge to assist answering the user's queries.\n{context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})


# ---- Azure Chat Model Example for Streaming (Reusable LLM Instance) ----

def get_azure_chat_openai():
    return AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        openai_api_type="azure",
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        streaming=True,
        temperature=0.3,
        # model="gpt-4o", # Optionally specify if you want (should match Azure deployment)
    )
