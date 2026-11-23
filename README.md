# 🎀 Bunny PDF Chatbot

An AI-powered Multi-PDF Chatbot built using **LangChain, Google Gemini, FAISS and Streamlit**. Upload multiple PDF documents and ask questions in natural language. The chatbot retrieves relevant information from your documents and generates contextual answers using Gemini.

## ✨ Features

- 📄 Upload and manage multiple PDF files.
- 💬 Ask questions about uploaded documents.
- 🤖 AI-powered responses using Google Gemini.
- 🔍 Semantic search using FAISS vector database.
- 🧠 Conversational memory for follow-up questions.
- 📚 Retrieve relevant document sources with page information.
- 🎀 Kawaii pastel-themed Streamlit interface.
- ⚡ Efficient document chunking and vector embeddings.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Frontend and web interface |
| LangChain | LLM application framework |
| Google Gemini | AI responses and embeddings |
| FAISS | Vector database |
| PyPDF | PDF text extraction |

## 📁 Project Structure

```text
Bunny-PDF-Chatbot/
│
├── app.py
├── .env                 # Local API keys (not uploaded)
├── .gitignore
├── requirements.txt
├── data/                # Uploaded PDF documents
└── faiss_index/         # Generated vector database
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ANISHA-KASHYAP/Bunny-PDF-Chatbot.git
cd Bunny-PDF-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

**Never upload your `.env` file or expose your API key on GitHub.**

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🚀 How It Works

1. Upload one or more PDF documents.
2. Extract text from the uploaded PDFs.
3. Split the text into smaller chunks.
4. Generate embeddings using Google Gemini.
5. Store embeddings in the FAISS vector database.
6. Retrieve relevant chunks based on the user's question.
7. Generate contextual answers using Gemini.

## 🔮 Future Improvements

- Support for more document formats.
- Improved source citations.
- Persistent chat history.
- Enhanced retrieval accuracy.
- Deployment for public access.

## 👩‍💻 Author

**Anisha Kashyap**

B.Tech CSE (Data Science)

- GitHub: [ANISHA-KASHYAP](https://github.com/ANISHA-KASHYAP)
- LinkedIn: [Anisha Kashyap](https://www.linkedin.com/in/anisha-kashyap-444995323/)

---

⭐ If you find this project useful, consider giving it a star!
