import streamlit as st
import os
import dotenv
import uuid

# Environment setup (required for Streamlit Cloud if using Linux)
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic  # If you want Anthropic fallback
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

# Load environment variables from .env file and override existing
dotenv.load_dotenv(override=True)

# REQUIRED: Set these ENV VARIABLES in .env or OS for Azure OpenAI
# AZURE_OPENAI_API_KEY
# AZURE_OPENAI_ENDPOINT
# AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
# OPENAI_API_VERSION
# OPENAI_API_TYPE=azure

# --- Model selection ---

MODELS = [
    "azure-openai/gpt-4o"  # Or whatever deployment name you provided in Azure
    # Add more if you create multiple deployments (optional)
]

# Application page config
st.set_page_config(
    page_title="RAG LLM app?",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- HEADER ---
st.html("""<h2 style="text-align: center;">📚🔍 Welcome to my Azure OpenAi's gpt4o, Embedding ChatBot </i> 🤖💬</h2>""")

# --- LLM SETUP ---
def get_azure_llm():
    return AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        openai_api_type="azure",
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        streaming=True,
        temperature=0.3,
    )

# Optionally add Anthropic Claude fallback if you wish:
# def get_anthropic_llm():
#     return ChatAnthropic(
#         # Set up with Anthropic API key, model, etc.
#     )

# --- SESSION & STATE INITIALIZATION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
]

# You can add more state variables as needed for docs, URLs, etc.
with st.sidebar:
    if "AZURE_OPENAI_API_KEY" not in os.environ:
        default_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY") if os.getenv("AZURE_OPENAI_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
        with st.popover("🔐 Azure OpenAI"):
            openai_api_key = st.text_input(
                "Introduce your OpenAI API Key (https://ai-openai-live-swc-01.openai.azure.com)", 
                value=default_openai_api_key, 
                type="password",
                key="azure_openai_api_key",
            )

        # default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") if os.getenv("ANTHROPIC_API_KEY") is not None else ""
        # with st.popover("🔐 Anthropic"):
        #     anthropic_api_key = st.text_input(
        #         "Introduce your Anthropic API Key (https://console.anthropic.com/)", 
        #         value=default_anthropic_api_key, 
        #         type="password",
        #         key="anthropic_api_key",
        #     )
    else:
        openai_api_key, anthropic_api_key = None, None
        st.session_state.openai_api_key = None
        az_openai_api_key = os.getenv("AZ_OPENAI_API_KEY")
        st.session_state.az_openai_api_key = az_openai_api_key



# --- DOCUMENT/URL LOAD BUTTONS ---
if st.button("Load Document(s)"):
    load_doc_to_db()
if st.button("Load Web Page"):
    load_url_to_db()

# --- Chat Interface ---
# st.markdown("#### Chat with your knowledge base")

llm = get_azure_llm()
user_message = st.text_input("You:", key="user_input", value="")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Stream response from retrieved context, if using RAG
    with st.spinner("Thinking..."):
        for chunk in stream_llm_rag_response(llm, st.session_state.messages):
            st.write(chunk, end="")

    # Alternatively, if you just want direct streaming response:
    # for chunk in stream_llm_response(llm, st.session_state.messages):
    #     st.write(chunk, end="")
    missing_openai = azure_openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key
# missing_anthropic = anthropic_api_key == "" or anthropic_api_key is None
# if missing_openai and missing_anthropic and ("AZURE_OPENAI_API_KEY" not in os.environ):
#     st.write("#")
#     st.warning("⬅️ Please introduce an API Key to continue...")

else:
    # Sidebar
    with st.sidebar:
        st.divider()
        models = []
        for model in MODELS:
            if "azure_openai" in model and not missing_openai:
                models.append(model)
            elif "anthropic" in model and not missing_anthropic:
                models.append(model)
            elif "azure-openai" in model:
                models.append(model)

        st.selectbox(
            "🤖 Select a Model", 
            options=models,
            key="model",
        )

        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "Use RAG", 
                value=is_vector_db_loaded, 
                key="use_rag", 
                disabled=not is_vector_db_loaded,
            )

        with cols0[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

        st.header("RAG Sources:")
            
        # File upload input for RAG with documents
        st.file_uploader(
            "📄 Upload a document", 
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
        )

        # URL input for RAG with websites
        st.text_input(
            "🌐 Introduce a URL", 
            placeholder="https://example.com",
            on_change=load_url_to_db,
            key="rag_url",
        )

        with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

    
    # Main chat app
    model_provider = st.session_state.model.split("/")[0]
    if model_provider == "Azure OpenAI":
        llm_stream = ChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    # elif model_provider == "anthropic":
    #     llm_stream = ChatAnthropic(
    #         api_key=anthropic_api_key,
    #         model=st.session_state.model.split("/")[-1],
    #         temperature=0.3,
    #         streaming=True,
    #     )
    elif model_provider == "azure-openai":
        llm_stream = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version="2025-01-01-preview",
            model_name=st.session_state.model.split("/")[-1],
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_type="azure",
            temperature=0.3,
            streaming=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

            if not st.session_state.use_rag:
                st.write_stream(stream_llm_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))


with st.sidebar:
    st.divider()
    st.video("https://youtu.be/abMwFViFFhI")
    st.write("📋[Medium Blog](https://medium.com/@enricdomingo/program-a-rag-llm-chat-app-with-langchain-streamlit-o1-gtp-4o-and-claude-3-5-529f0f164a5e)")
    st.write("📋[GitHub Repo](https://github.com/enricd/rag_llm_app)")


# --- Conversation history ---
st.markdown("#### Conversation History")
for m in st.session_state.messages:
    st.write(f"{m['role'].capitalize()}: {m['content']}")

