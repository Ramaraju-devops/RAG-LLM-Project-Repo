# RAG LLM Chat Application 📚🤖

A comprehensive Retrieval Augmented Generation (RAG) application built with Python, LangChain, and Streamlit. This project demonstrates how to build an intelligent chatbot that can answer questions based on uploaded documents and web content using Azure OpenAI and OpenAI models.

# ---------
# Azure Services Required:
- Azure OpenAI
    - Model gpt4o
    - Embedded model - text-embedding-ada-002

# add the below keys in .env file
# Your Azure OpenAI API Key
AZURE_OPENAI_API_KEY="Enter Azure OpenAI Key"

# Your Azure OpenAI Endpoint (replace 'your-resource-name' with your actual resource name)
AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com"

# API Version (use the latest stable version)
AZURE_OPENAI_API_VERSION="2025-01-01-preview"

# Deployment names for your models (replace with your actual deployment names)
# AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="text-embedding-ada-002"

AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-4o"

AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME="text-embedding-ada-002"

# Optional: User Agent for requests
USER_AGENT="RAG_LLM_APP"

OPENAI_API_VERSION=2023-12-01-preview

# ---------

## 🌟 Features

- **Multi-Model Support**: Compatible with Azure OpenAI GPT-4o, OpenAI GPT models, and Anthropic Claude
- **Document Processing**: Upload and process PDF, DOCX, and text files
- **Web Content Integration**: Load and process content from URLs
- **Vector Database**: Uses ChromaDB for efficient document retrieval
- **Streaming Responses**: Real-time streaming of AI responses
- **Chat History**: Maintains conversation context for better interactions
- **Modern UI**: Clean and intuitive Streamlit interface

## 🏗️ Architecture

The application follows a standard RAG pipeline:

1. **Document Ingestion**: Load documents from files or URLs
2. **Text Splitting**: Break documents into manageable chunks
3. **Embedding**: Convert text chunks to vector embeddings
4. **Vector Storage**: Store embeddings in ChromaDB
5. **Retrieval**: Find relevant documents based on user queries
6. **Generation**: Generate responses using retrieved context

## 📋 Prerequisites

- Python 3.8 or higher
- Azure OpenAI account (for Azure models) or OpenAI API key
- Git

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RAG-LLM-Project-Repo
```

### 2. Create Virtual Environment

```bash
python -m venv ram_env1
# On Windows:
ram_env1\Scripts\activate
# On macOS/Linux:
source ram_env1/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory with the following variables:

#### For Azure OpenAI:
```env
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment_name
OPENAI_API_VERSION=2024-02-15-preview
OPENAI_API_TYPE=azure
```

#### For OpenAI:
```env
OPENAI_API_KEY=your_openai_api_key
```

#### For Anthropic (optional):
```env
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 5. Run the Application

#### Azure OpenAI Version:
```bash
streamlit run app.py
```

#### OpenAI Version:
```bash
streamlit run app-openai.py
```

## 📁 Project Structure

```
RAG-LLM-Project-Repo/
├── app.py                      # Main Streamlit app (Azure OpenAI)
├── app-openai.py              # Streamlit app (OpenAI)
├── app1.py                    # Alternative app version
├── rag_methods.py             # Core RAG functionality
├── rag_methods-openai.py      # OpenAI-specific RAG methods
├── azure_rag_config.py        # Azure configuration utilities
├── azure_notebook_cells.py    # Jupyter notebook cells for Azure
├── rag_lanchain.ipynb         # Jupyter notebook with examples
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── .gitignore                # Git ignore rules
├── README_Azure_Setup.md      # Azure setup instructions
├── docs/                      # Sample documents
│   ├── test_rag.pdf
│   └── test_rag.docx
└── source_files/              # Directory for uploaded files
```

## 🔧 Configuration

### Model Selection

The application supports multiple AI models:

- **Azure OpenAI**: GPT-4o, GPT-4, GPT-3.5-turbo
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3.5-Sonnet

### Document Types Supported

- PDF files (`.pdf`)
- Word documents (`.docx`)
- Text files (`.txt`)
- Web pages (via URL)

## 💡 Usage

1. **Start the Application**: Run the Streamlit app using the command above
2. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or text files
3. **Add Web Content**: Enter URLs to load web content into the knowledge base
4. **Ask Questions**: Type your questions in the chat interface
5. **Get AI Responses**: Receive contextual answers based on your uploaded content

## 🛠️ Development

### Key Components

- **`rag_methods.py`**: Contains core RAG functionality including document loading, text splitting, embedding, and retrieval
- **`app.py`**: Main Streamlit application with UI components
- **`azure_rag_config.py`**: Azure-specific configurations and utilities

### Customization

- **Chunk Size**: Modify text splitting parameters in `rag_methods.py`
- **Model Parameters**: Adjust temperature, max tokens, etc. in the app files
- **UI Components**: Customize the Streamlit interface in the main app files

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `requirements.txt`
2. **API Key Issues**: Verify your API keys are correctly set in the `.env` file
3. **ChromaDB Issues**: On Linux systems, the app automatically handles SQLite compatibility
4. **Memory Issues**: Large documents may require chunking adjustments

### Linux/Streamlit Cloud Compatibility

The application includes automatic SQLite compatibility fixes for Linux environments:

```python
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

## 📚 Learning Resources

- **Jupyter Notebook**: `rag_lanchain.ipynb` contains step-by-step examples
- **Azure Setup**: `README_Azure_Setup.md` provides detailed Azure configuration
- **Sample Documents**: Use files in the `docs/` folder for testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web interface
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) and [OpenAI](https://openai.com/) for language models

## 📞 Support

If you encounter any issues or have questions, please:

1. Check the troubleshooting section above
2. Review the Azure setup guide if using Azure OpenAI
3. Open an issue on GitHub with detailed information about your problem

---

**Happy RAG Building! 🚀**
