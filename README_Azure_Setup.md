# Azure OpenAI RAG Setup Guide

This guide shows how to configure your RAG application to use Azure OpenAI instead of regular OpenAI.

## Prerequisites

1. **Azure OpenAI Service**: You need an Azure OpenAI resource deployed in Azure
2. **Model Deployments**: Deploy the following models in your Azure OpenAI resource:
   - **Chat Model**: GPT-4o, GPT-4, or GPT-3.5-turbo
   - **Embeddings Model**: text-embedding-ada-002 or text-embedding-3-small

## Setup Steps

### 1. Get Your Azure OpenAI Credentials

From your Azure OpenAI resource in the Azure portal, collect:
- **API Key**: Found in "Keys and Endpoint" section
- **Endpoint**: Your resource endpoint (e.g., `https://your-resource-name.openai.azure.com/`)
- **Deployment Names**: Names of your deployed models

### 2. Configure Environment Variables

Copy the example environment file:
```bash
cp .env.azure.example .env
```

Edit `.env` file with your actual values:
```env
AZURE_OPENAI_API_KEY=your_actual_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your-chat-deployment-name
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=your-embeddings-deployment-name
```

### 3. Install Required Packages

Make sure you have the required packages installed:
```bash
pip install langchain-openai langchain-community langchain-anthropic chromadb
```

### 4. Update Your Notebook

Replace the OpenAI imports and configuration in your notebook with Azure OpenAI equivalents.

#### Key Changes:

**Old (OpenAI):**
```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()
```

**New (Azure OpenAI):**
```python
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.7
)
```

## Files Created

- `azure_rag_config.py`: Complete working example with Azure OpenAI
- `.env.azure.example`: Template for environment variables
- `azure_notebook_cells.py`: Notebook cells you can copy-paste
- `README_Azure_Setup.md`: This setup guide

## Usage Examples

### Option 1: Use the Complete Script
```bash
python azure_rag_config.py
```

### Option 2: Copy Cells to Your Notebook
Open `azure_notebook_cells.py` and copy the cell contents into your Jupyter notebook.

### Option 3: Modify Your Existing Notebook
1. Replace the imports in your first cell
2. Replace the embeddings and LLM configuration
3. Keep everything else the same

## Testing Your Setup

After configuration, test with these questions:
- "What is my favorite food?"
- "How many bottles are in the truck?"
- "What's new in Streamlit's latest version?"

## Troubleshooting

### Common Issues:

1. **Authentication Error**: Check your API key and endpoint
2. **Deployment Not Found**: Verify your deployment names match exactly
3. **API Version Error**: Use a supported API version (2024-02-15-preview is recommended)
4. **Rate Limits**: Azure OpenAI has different rate limits than regular OpenAI

### Checking Your Configuration:
```python
# Test embeddings
test_embedding = embeddings.embed_query("test")
print(f"Embedding dimension: {len(test_embedding)}")

# Test chat model
test_response = llm.invoke("Hello, how are you?")
print(f"Chat response: {test_response.content}")
```

## Benefits of Azure OpenAI

- **Enterprise Security**: Better compliance and security features
- **Data Privacy**: Your data stays in your Azure tenant
- **SLA**: Enterprise-grade service level agreements
- **Integration**: Better integration with other Azure services
- **Cost Control**: More predictable pricing and budgeting

## Next Steps

Once your Azure OpenAI setup is working:
1. Experiment with different model deployments
2. Adjust chunk sizes and overlap for better retrieval
3. Customize the system prompts for your use case
4. Add more document types and sources
5. Implement conversation memory persistence
