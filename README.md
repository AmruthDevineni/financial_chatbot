# financial_chatbot

# Financial Chatbot using SEC Filings, LLMs, and Pinecone

This is a financial chatbot built using **LangChain**, **FinBERT embeddings**, **SEC EDGAR filings**, and **Streamlit**. The app allows users to ask detailed questions about companies using 10-K reports, and receive answers grounded in real documents with source traceability.

---

## Key Features

- Ingests SEC 10-K filings via Edgar API
-  Chunks and embeds documents using FinBERT
-  Stores embeddings in Pinecone vector DB
-  Answers financial questions using Ollama's LLaMA-2 model
-  Cites original sources in each response
-  Streamlit-based UI for easy use

---

## Project Structure
financial_chatbot 
  ├── app.py 
# Streamlit UI entrypoint 
  ├── chunk.py 
# Sentence-level chunking with metadata
  ├── finbert_embed.py 
# FinBERT-based embedding pipeline
  ├── llama_qa.py 
# Query-answer pipeline using Ollama + Pinecone 
  ├── edgar_client.py
# SEC EDGAR integration to fetch 10-Ks 
  ├── pinecone_utils.py 
# Pinecone setup and indexing logic 
  ├── constants.py
# Configuration and static variables
  ├── requirements.txt
# All dependencies 
  └── .env


---

## Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/AmruthDevineni/financial_chatbot.git
cd financial_chatbot
pip install -r requirements.txt


PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_pinecone_region
OLLAMA_BASE_URL=http://localhost:11434
SEC_API_KEY=your_sec_api_key_if_required


ollama run llama2
streamlit run app.py
```

## 🧠 How It Works

### `chunk.py`
- Splits SEC filings into **clean, sentence-based chunks**.
- Adds metadata (section, line number) for traceability.

### `finbert_embed.py`
- Uses FinBERT from HuggingFace (`yiyanghkust/finbert-tone`) to embed each chunk.
- Embeddings are sent to Pinecone via `pinecone_utils.py`.

### `llama_qa.py`
- Given a user question:
  1. Query is embedded.
  2. Top-k similar chunks from Pinecone are retrieved.
  3. Prompt is built: question + retrieved context.
  4. Sent to LLaMA via Ollama API for grounded response.
  5. Output includes answer + citations.

### `app.py`
- Provides **Streamlit interface** with:
  - Input box for financial queries
  - Dropdown to select company/filing
  - Display of answer and original source
  - Optional sidebar with metadata

---

## 💡 Example Queries

- “What was Apple's net income in 2021?”
- “How did Tesla describe supply chain risks?”
- “Break down Microsoft's revenue streams last fiscal year.”

---

## 📦 Dependencies

Major packages:

- `langchain`
- `transformers`
- `streamlit`
- `pinecone-client`
- `openai` or `ollama`
- `sec-edgar-downloader` or custom `edgar_client.py`

Install them all via:

```bash
pip install -r requirements.txt
```
## 📌 Future Improvements

- Upload user PDFs of 10-Ks directly
- Add charts for revenue/net income over time
- Integrate evaluation tools for answer quality
- Expand to multi-company comparisons

---

## 🙋‍♂️ Acknowledgements

- FinBERT by ProsusAI
- [LangChain](https://www.langchain.com/)
- [Pinecone](https://www.pinecone.io/)
- [SEC EDGAR](https://www.sec.gov/edgar.shtml)
- [Ollama](https://ollama.com)




