"""
Azure OpenAI Configuration for RAG Application
This file shows how to configure the RAG application to use Azure OpenAI instead of regular OpenAI.
"""

import os
import dotenv
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import Azure OpenAI classes instead of regular OpenAI
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
dotenv.load_dotenv()

def setup_azure_openai():
    """
    Configure Azure OpenAI settings
    You need to set these environment variables in your .env file:
    
    AZURE_OPENAI_API_KEY=your_azure_openai_api_key
    AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
    AZURE_OPENAI_API_VERSION=2024-02-15-preview
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your-chat-model-deployment-name
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=your-embeddings-model-deployment-name
    """
    
    # Azure OpenAI Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    
    # Azure OpenAI Chat Model
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.7
    )
    
    return embeddings, llm

def load_documents():
    """Load documents from local files and web URLs"""
    doc_paths = [
        "./docs/test_rag.docx",
        "./docs/test_rag.pdf",
    ]
    
    docs = [] 
    for doc_file in doc_paths:
        file_path = Path(doc_file)
        
        try:
            if doc_file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif doc_file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif doc_file.endswith(".txt") or doc_file.endswith(".md"):
                loader = TextLoader(file_path)
            else:
                print(f"Document type {doc_file} not supported.")
                continue
            
            docs.extend(loader.load())
            
        except Exception as e:
            print(f"Error loading document {doc_file}: {e}")
    
    # Load URLs
    url = "https://docs.streamlit.io/develop/quick-reference/release-notes"
    try:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    except Exception as e:
        print(f"Error loading document from {url}: {e}")
    
    return docs

def create_vector_store(docs, embeddings):
    """Create vector store with document chunks"""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )
    
    document_chunks = text_splitter.split_documents(docs)
    
    # Create vector store with Azure OpenAI embeddings
    vector_db = Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
    )
    
    return vector_db

def create_rag_chain(vector_db, llm):
    """Create the complete RAG chain"""
    
    # Create retriever
    retriever = vector_db.as_retriever()
    
    # Create contextualize prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Create QA prompt
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create document chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def main():
    """Main function to set up and test the RAG system"""
    print("Setting up Azure OpenAI RAG system...")
    
    # Setup Azure OpenAI
    embeddings, llm = setup_azure_openai()
    
    # Load documents
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    
    # Create vector store
    print("Creating vector store...")
    vector_db = create_vector_store(docs, embeddings)
    
    # Create RAG chain
    print("Creating RAG chain...")
    rag_chain = create_rag_chain(vector_db, llm)
    
    print("RAG system ready!")
    
    # Test the system
    chat_history = []
    
    # Example query
    result = rag_chain.invoke({
        "input": "What is my favorite food?",
        "chat_history": chat_history
    })
    
    print(f"\nQuestion: What is my favorite food?")
    print(f"Answer: {result['answer']}")
    
    return rag_chain

if __name__ == "__main__":
    main()
