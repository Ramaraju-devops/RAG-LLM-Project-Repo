"""
Azure OpenAI Notebook Cells
Copy these cells into your Jupyter notebook to replace the OpenAI configuration
"""

# Cell 1: Imports and Setup
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

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

dotenv.load_dotenv()
"""

# Cell 2: Azure OpenAI Configuration
"""
# Configure Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Configure Azure OpenAI Chat Model
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.7
)

print("Azure OpenAI configured successfully!")
"""

# Cell 3: Load Documents (keep this the same)
"""
# Load docs
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

print(f"Loaded {len(docs)} documents")
"""

# Cell 4: Split Documents (keep this the same)
"""
# Split docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=1000,
)

document_chunks = text_splitter.split_documents(docs)
print(f"Created {len(document_chunks)} document chunks")
"""

# Cell 5: Create Vector Store with Azure OpenAI Embeddings
"""
# Create vector store with Azure OpenAI embeddings
vector_db = Chroma.from_documents(
    documents=document_chunks,
    embedding=embeddings,  # Using Azure OpenAI embeddings
)

print("Vector store created successfully!")
"""

# Cell 6: Create RAG Chain
"""
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
    "\\n\\n"
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

print("RAG chain created successfully!")
"""

# Cell 7: Test the RAG System
"""
# Test the system
chat_history = []

# Example query
result = rag_chain.invoke({
    "input": "What is my favorite food?",
    "chat_history": chat_history
})

print(f"Question: What is my favorite food?")
print(f"Answer: {result['answer']}")

# Add to chat history for next question
chat_history.extend([
    HumanMessage(content="What is my favorite food?"),
    AIMessage(content=result["answer"])
])
"""

# Cell 8: Interactive Chat Function
"""
def chat_with_rag(question, chat_history=[]):
    \"\"\"Function to interact with the RAG system\"\"\"
    result = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print("-" * 50)
    
    # Update chat history
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=result["answer"])
    ])
    
    return result, chat_history

# Example usage:
# result, history = chat_with_rag("How many bottles are in the truck?")
# result, history = chat_with_rag("What about Streamlit's latest version?", history)
"""
