<h1 align="center"> Retrieval-Augmented Generation with Gradio and Groq API Key</h1>
<p align="center"> Natural Language Processing Project</p>

<div align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
</div>

### Name : Muhammad Rahmananda Arief Wibisono  
### Tech Stack : Python, Gradio, LangChain, HuggingFace Embedding, FAISS vector store

---

### 1. Analysis about how the project works

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and Groq API. The general workflow is as follows:

1. PDF documents are loaded using `PyPDFLoader`.
2. Texts are split into smaller chunks using `RecursiveCharacterTextSplitter`.
3. Each chunk is embedded using `HuggingFaceEmbeddings`.
4. The embeddings are stored and searched using FAISS (vector database).
5. When a user asks a question, similar text chunks are retrieved from FAISS.
6. The selected model from Groq (e.g. LLaMA 3.3-70B) generates a final answer based on the retrieved context.
7. The UI is built with Gradio to allow easy interaction with the system.

---

### 2. Analysis about how different every model works on Retrieval-Augmented Generation

```python
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile", # Change the model in the code
        temperature=0.2
    )
```
- Model used : ```[llama-3.3-70b-versatile, deepseek-r1-distill-llama-70b, gemma2-9b-it]```

2.1 Analysis on ```llama-3.3-70b-versatile``` : 
- Most contextually accurate and fluent.
- Best for long and complex inputs.
- Slightly slower response time compared to smaller models.

2.2 Analysis on ```deepseek-r1-distill-llama-70b``` : 
- Distilled version with faster inference.
- Slight trade-off in depth and nuance of answers.
- Good balance between speed and quality.

2.3 Analysis on ```gemma2-9b-it``` : 
- Lightweight, very fast inference.
- Suitable for simple Q&A.
- May lose contextual accuracy on dense or technical documents.

### 3. Analysis about how temperature works

```python
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2 # Change the temperature value here and analzye
    )
```

3.1 Analysis on higher temperature 
- Produces more creative and varied answers.
- Can introduce hallucinations or go off-topic.
- Useful for brainstorming or idea generation.

3.2 Analysis on lower temperature
- Deterministic and factual responses.
- Maintains consistency across repeated queries.
- Ideal for academic, legal, or research-based answers.

### 4. How to run the project

# Step 1: Clone this repository
git clone https://github.com/arifian853/RAG_with_GroqAPI.git
cd RAG_with_GroqAPI

# Step 2: Setup .env file
cp .env.example .env
# Then open .env and paste your Groq API Key
GROQ_API_KEY=your-groq-api-key

# Step 3: Install all dependencies
pip install -r requirements.txt

# Step 4: Create and activate virtual environment
python -m venv venv
venv\Scripts\activate    # On Windows

# Step 5: Run the application
python app.py

## notes
- If Gradio fails to share the link publicly, ensure your Windows Defender or firewall is not blocking it.
- Use a stable internet connection to access Groq API.
```
