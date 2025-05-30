
# Conversational Website Q&A Bot

This assistant scrapes content from a given website, stores it in a vector database, and uses OpenAI's language models to answer questions based on the scraped content. It supports both English and Arabic, and offers a real-time chat experience using Streamlit.

---

## 🧠 How It Works

### Web Scraping
- Uses `WebBaseLoader` from LangChain to load content from a provided URL (e.g., http://localhost:5173/).

### Chunking and Embedding
- Text is split into manageable chunks using `RecursiveCharacterTextSplitter`.
- Chunks are embedded using `OpenAIEmbeddings`.

### Vector Storage with Chroma
- Embeddings are stored in Chroma for fast semantic retrieval.

### Conversation-Aware Retrieval
- A retriever chain is used to understand the context of the conversation.
- A RAG chain retrieves relevant chunks and uses `ChatOpenAI` to generate context-aware responses.

### Chat Interface
- Built with Streamlit’s `chat_input` and `chat_message` components.
- Fully interactive user experience.

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/invasion-assistant.git
cd invasion-assistant
```

### 2. Install Required Dependencies
Make sure you have Python 3.9+ installed. Then install dependencies:

```bash
pip install -r requirements.txt
```

#### Example `requirements.txt`:
```
streamlit
langchain
openai
python-dotenv
chromadb
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Start the Website to be Indexed
Make sure your local website (e.g., http://localhost:5173/) is running.
Alternatively, replace the `website_url` in the code with any public site.

### 5. Launch the App
```bash
streamlit run app.py
```
*Replace `app.py` with your filename if different.*

---

## 💬 Using the Assistant
- The assistant will greet you and load content from the configured site.
- Type your questions in English or Arabic.
- The assistant will respond with context-based answers using the indexed content.

---

## ✅ Example Questions
- "ما هي الأفكار الرئيسية المعروضة في الموقع؟"
- "هل هناك حلول مبتكرة تم ذكرها؟"
- "What is this project about?"
- "Explain the purpose of this website."

---

## 📦 Future Improvements
- Support for multiple URLs.
- UI enhancements (theme switch, avatars, etc.).
- Chat memory persistence across sessions.
- Multilingual summarization.

---

## 🛠 Built With
- Streamlit  
- LangChain  
- Chroma Vector DB  
- OpenAI API  
- dotenv

---

## 📄 License
MIT License – feel free to use and adapt!
