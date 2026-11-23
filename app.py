
import os
import glob
import time
import shutil
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

st.set_page_config(
    page_title="Bunny PDF Chat ✨",
    page_icon="🎀",
    layout="wide"
)

DATA_FOLDER = "data"
INDEX_FOLDER = "faiss_index"

os.makedirs(DATA_FOLDER, exist_ok=True)


# ==========================================
# 2. KAWAII PASTEL CSS
# ==========================================

st.markdown("""
<style>

/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif !important;
}

.stApp {
    background: linear-gradient(
        135deg,
        #fff7fb 0%,
        #f7f0ff 50%,
        #f0f7ff 100%
    );
    color: #51445f;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Headings */
h1, h2, h3 {
    color: #76558f !important;
    font-weight: 900 !important;
}

/* Cute hero banner */
.hero-card {
    background: linear-gradient(
        120deg,
        #f6d9ec,
        #e7dcff,
        #dceeff
    );
    padding: 35px;
    border-radius: 28px;
    border: 2px solid white;
    box-shadow: 0 8px 25px rgba(154, 117, 180, 0.13);
    margin-bottom: 25px;
    text-align: center;
}

.hero-title {
    color: #684d83;
    font-size: 36px;
    font-weight: 900;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #806b91;
    font-size: 16px;
    font-weight: 600;
}

.hero-mascot {
    font-size: 55px;
    margin-bottom: 8px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #f8e8f4,
        #eee6ff,
        #e6f2ff
    );
    border-right: 2px solid #eadcf5;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #79568e !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        110deg,
        #e9b9db,
        #cdb9f4
    );
    color: #563d70 !important;
    border: 2px solid white !important;
    border-radius: 15px !important;
    font-weight: 800 !important;
    padding: 10px 18px !important;
    transition: all 0.25s ease;
    box-shadow: 0 4px 12px rgba(160, 120, 190, 0.13);
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 7px 17px rgba(160, 120, 190, 0.23);
    border-color: #d9b8f1 !important;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background: #fffaff !important;
    border: 1.5px solid #e8d4f3 !important;
    border-radius: 13px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #fffaff;
    border: 2px dashed #d8b9eb;
    border-radius: 18px;
    padding: 12px;
}

/* Chat message boxes */
[data-testid="stChatMessage"] {
    background: rgba(255, 250, 255, 0.92) !important;
    border: 1px solid #eadcf5;
    border-radius: 20px;
    padding: 15px;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(150, 120, 180, 0.06);
}

/* ==========================================
   FORCE BLACK TEXT IN CHAT OUTPUT
   ========================================== */

/* AI and user message content */
[data-testid="stChatMessage"],
[data-testid="stChatMessage"] *,
[data-testid="stChatMessageContent"],
[data-testid="stChatMessageContent"] *,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
    color: #171717 !important;
}

/* Paragraphs and headings inside chat */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] blockquote {
    color: #171717 !important;
}

/* Chat tables */
[data-testid="stChatMessage"] table,
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] td {
    color: #171717 !important;
}

/* Chat code blocks */
[data-testid="stChatMessage"] code,
[data-testid="stChatMessage"] pre {
    color: #171717 !important;
}

/* Chat input text */
[data-testid="stChatInput"] textarea {
    color: #171717 !important;
    caret-color: #171717 !important;
}

/* Chat input placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #8c7b96 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #fffaff;
    border: 1.5px solid #e8d8f1;
    border-radius: 16px;
}

/* Status messages */
.stAlert {
    border-radius: 15px;
}

/* Dividers */
hr {
    border-color: #e8d8f1 !important;
}

/* Cute footer */
.footer {
    text-align: center;
    color: #9a83aa;
    font-size: 13px;
    padding: 25px 0 5px;
    font-weight: 700;
}

/* Hide default Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. HERO SECTION
# ==========================================

st.markdown("""
<div class="hero-card">
    <div class="hero-mascot">🐰📚✨</div>
    <div class="hero-title">Bunny PDF Chat</div>
    <div class="hero-subtitle">
        Your little AI study buddy! 💗
        <br>
        Upload your PDFs, ask questions and learn something new.
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 4. API KEY
# ==========================================

api_key = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not api_key:
    st.error("Google API key not found in .env file.")
    st.stop()


# ==========================================
# 5. INITIALIZE MODELS
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
    google_api_key=api_key
)


# ==========================================
# 6. DATABASE FUNCTIONS
# ==========================================

def invalidate_database():

    if os.path.exists(INDEX_FOLDER):
        shutil.rmtree(INDEX_FOLDER)

    st.session_state.vector_store = None
    st.session_state.messages = []


# ==========================================
# 7. UPLOAD PDFS
# ==========================================

def save_uploaded_pdfs(uploaded_files):

    saved_count = 0

    for uploaded_file in uploaded_files:

        filename = os.path.basename(uploaded_file.name)

        file_path = os.path.join(
            DATA_FOLDER,
            filename
        )

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        saved_count += 1

    if saved_count:
        invalidate_database()

    return saved_count


# ==========================================
# 8. DELETE PDFS
# ==========================================

def delete_pdfs(selected_files):

    deleted_count = 0

    for filename in selected_files:

        file_path = os.path.join(
            DATA_FOLDER,
            filename
        )

        if os.path.isfile(file_path):
            os.remove(file_path)
            deleted_count += 1

    if deleted_count:
        invalidate_database()

    return deleted_count


# ==========================================
# 9. LOAD PDFS
# ==========================================

def load_pdfs():

    pdf_files = glob.glob(
        os.path.join(DATA_FOLDER, "*.pdf")
    )

    if not pdf_files:
        raise ValueError("No PDFs found in data folder.")

    all_documents = []

    for pdf_file in pdf_files:

        loader = PyPDFLoader(pdf_file)

        documents = loader.load()

        all_documents.extend(documents)

    return all_documents


# ==========================================
# 10. SPLIT DOCUMENTS
# ==========================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)


# ==========================================
# 11. CREATE FAISS DATABASE
# ==========================================

def create_vector_store(chunks):

    batch_size = 10
    vector_store = None

    total_batches = (
        len(chunks) + batch_size - 1
    ) // batch_size

    progress = st.progress(0)
    status = st.empty()

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i + batch_size]

        current_batch = (i // batch_size) + 1

        status.info(
            f"🌸 Creating embeddings: "
            f"{current_batch}/{total_batches}"
        )

        for attempt in range(3):

            try:

                batch_store = FAISS.from_documents(
                    documents=batch,
                    embedding=embeddings
                )

                break

            except Exception as e:

                error = str(e)

                if "429" in error or "RESOURCE_EXHAUSTED" in error:

                    if attempt == 2:
                        raise

                    status.warning(
                        "API quota reached. Waiting 30 seconds..."
                    )

                    time.sleep(30)

                else:
                    raise

        if vector_store is None:

            vector_store = batch_store

        else:

            vector_store.merge_from(batch_store)

        progress.progress(
            current_batch / total_batches
        )

        if i + batch_size < len(chunks):
            time.sleep(7.5)

    status.success("✨ Embeddings created successfully!")

    return vector_store


# ==========================================
# 12. SAVE FAISS DATABASE
# ==========================================

def save_vector_store(vector_store):

    os.makedirs(
        INDEX_FOLDER,
        exist_ok=True
    )

    vector_store.save_local(INDEX_FOLDER)


# ==========================================
# 13. LOAD SAVED FAISS DATABASE
# ==========================================

def load_saved_vector_store():

    index_file = os.path.join(
        INDEX_FOLDER,
        "index.faiss"
    )

    pickle_file = os.path.join(
        INDEX_FOLDER,
        "index.pkl"
    )

    if (
        os.path.exists(index_file)
        and os.path.exists(pickle_file)
    ):

        return FAISS.load_local(
            INDEX_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True
        )

    return None


# ==========================================
# 14. CONVERSATIONAL MEMORY
# ==========================================

def get_standalone_question(question, chat_history):

    if not chat_history:
        return question

    history_text = ""

    for message in chat_history[-8:]:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    rewrite_prompt = f"""
You are a question contextualization assistant.

Your task is to rewrite the latest user question
as a standalone question using the conversation history.

Rules:
1. Resolve references like it, its, they, them and this.
2. Understand the topic from previous messages.
3. Do not answer the question.
4. Do not add new information.
5. If the question is already clear, return it unchanged.
6. Return ONLY the rewritten question.

CONVERSATION HISTORY:
{history_text}

LATEST QUESTION:
{question}

STANDALONE QUESTION:
"""

    response = llm.invoke(rewrite_prompt)

    return str(response.content).strip()


# ==========================================
# 15. SESSION STATE
# ==========================================

if "vector_store" not in st.session_state:

    st.session_state.vector_store = None

    try:

        st.session_state.vector_store = (
            load_saved_vector_store()
        )

    except Exception as e:

        st.warning(
            f"Could not load saved database: {e}"
        )

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 16. SIDEBAR - DOCUMENT MANAGER
# ==========================================

with st.sidebar:

    st.markdown(
        "## 🎀 Your Little Library"
    )

    st.caption("Manage your PDF collection here! 💕")

    st.divider()

    # PDF UPLOAD

    st.markdown("### 📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose your PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "💗 Save My PDFs",
        use_container_width=True
    ):

        if uploaded_files:

            try:

                count = save_uploaded_pdfs(
                    uploaded_files
                )

                st.success(
                    f"Yay! {count} PDF(s) uploaded! 🎉"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Upload failed: {str(e)}"
                )

        else:

            st.warning("Choose some PDFs first! 🥺")

    st.divider()

    # DISPLAY DOCUMENTS

    st.markdown("### 📚 My Documents")

    pdf_files = sorted([
        os.path.basename(file)
        for file in glob.glob(
            os.path.join(DATA_FOLDER, "*.pdf")
        )
    ])

    st.markdown(
        f"**📖 Total PDFs: {len(pdf_files)}**"
    )

    if pdf_files:

        for filename in pdf_files:

            st.markdown(
                f"📄 `{filename}`"
            )

    else:

        st.info("Your library is waiting for PDFs! 🌷")

    st.divider()

    # DELETE DOCUMENTS

    st.markdown("### 🗑️ Clean My Library")

    if pdf_files:

        selected_files = st.multiselect(
            "Select PDFs to delete",
            options=pdf_files
        )

        if st.button(
            "🗑️ Delete Selected PDFs",
            use_container_width=True
        ):

            if selected_files:

                count = delete_pdfs(selected_files)

                st.success(
                    f"{count} PDF(s) deleted!"
                )

                st.rerun()

            else:

                st.warning("Select at least one PDF.")

    st.divider()

    # DATABASE STATUS

    st.markdown("### 🌸 AI Database Status")

    if st.session_state.vector_store is not None:

        st.success("✨ Your AI library is ready!")

    else:

        st.info("Your AI library needs processing.")

    # PROCESS DOCUMENTS

    if st.button(
        "✨ Process / Rebuild PDFs",
        use_container_width=True
    ):

        try:

            with st.spinner("Reading your PDFs... 📖"):

                documents = load_pdfs()

                if not any(
                    doc.page_content.strip()
                    for doc in documents
                ):

                    raise ValueError(
                        "No readable text found in PDFs."
                    )

                chunks = split_documents(documents)

                if not chunks:
                    raise ValueError("No chunks generated.")

            st.write(f"Total chunks: {len(chunks)}")

            with st.spinner("Building your AI library... 💗"):

                new_vector_store = create_vector_store(chunks)

                save_vector_store(new_vector_store)

                st.session_state.vector_store = new_vector_store

                st.session_state.messages = []

            st.success(
                "Your PDFs are ready to chat! 🎉"
            )

        except Exception as e:

            st.error(
                f"Processing failed: {str(e)}"
            )

    st.divider()

    # CLEAR CHAT

    if st.button(
        "🧹 Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.markdown(
        '<div class="footer">Made with 💗 and a little magic</div>',
        unsafe_allow_html=True
    )


# ==========================================
# 17. MAIN CHAT AREA
# ==========================================

st.markdown("## 💬 Let's Chat!")

st.caption(
    "Ask questions from your uploaded PDFs. "
    "Your little AI buddy is here to help! 🐰"
)

if st.session_state.vector_store is None:

    st.markdown("""
    <div class="hero-card">
        <div class="hero-mascot">🥺📚</div>
        <div class="hero-title" style="font-size:25px;">
            Your library is empty!
        </div>
        <div class="hero-subtitle">
            Upload your PDF documents from the sidebar
            and process them to start chatting. 💕
        </div>
    </div>
    """, unsafe_allow_html=True)

else:

    # WELCOME PROMPTS

    if not st.session_state.messages:

        st.markdown("### 🌷 Try asking something!")

        col1, col2 = st.columns(2)

        with col1:

            st.info(
                "📖 Explain the main concepts in my PDFs."
            )

            st.info(
                "📝 Give me a summary of the documents."
            )

        with col2:

            st.info(
                "🎯 Create important exam questions."
            )

            st.info(
                "💡 Explain a topic with examples."
            )

    # DISPLAY CHAT HISTORY

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # CHAT INPUT

    question = st.chat_input(
        "Type your question here... 💗"
    )

    if question:

        previous_history = (
            st.session_state.messages.copy()
        )

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                with st.spinner("Thinking... 🐰"):

                    standalone_question = (
                        get_standalone_question(
                            question,
                            previous_history
                        )
                    )

                with st.spinner("Searching your PDFs... 📚"):

                    retriever = (
                        st.session_state.vector_store
                        .as_retriever(
                            search_kwargs={"k": 4}
                        )
                    )

                    relevant_docs = retriever.invoke(
                        standalone_question
                    )

                    context = "\n\n".join(
                        [
                            f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}\n"
                            f"Page: {doc.metadata.get('page', 0) + 1}\n"
                            f"{doc.page_content}"
                            for doc in relevant_docs
                        ]
                    )

                    history_text = ""

                    for message in previous_history[-8:]:

                        history_text += (
                            f"{message['role']}: "
                            f"{message['content']}\n"
                        )

                    answer_prompt = f"""
You are an intelligent assistant that answers
questions using multiple PDF documents.

Instructions:
1. Answer using the provided PDF context.
2. Do not invent facts.
3. If information is unavailable in the PDFs,
   clearly say that it was not found.
4. Use conversation history to understand follow-up questions.
5. Give clear, structured and beginner-friendly answers.
6. Mention relevant PDF sources.

CONVERSATION HISTORY:
{history_text}

REWRITTEN QUESTION:
{standalone_question}

PDF CONTEXT:
{context}

LATEST USER QUESTION:
{question}

ANSWER:
"""

                    response = llm.invoke(answer_prompt)

                    answer = response.content

                    if isinstance(answer, list):

                        answer = "\n".join(
                            item.get("text", "")
                            for item in answer
                            if isinstance(item, dict)
                        )

                    answer = str(answer)

                    st.markdown(answer)

                    # SOURCE DISPLAY

                    with st.expander("📖 View PDF Sources"):

                        for doc in relevant_docs:

                            filename = os.path.basename(
                                doc.metadata.get(
                                    "source", "Unknown"
                                )
                            )

                            page = (
                                doc.metadata.get("page", 0) + 1
                            )

                            st.write(
                                f"📄 {filename} — Page {page}"
                            )

                    # SAVE RESPONSE

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

            except Exception as e:

                st.error(
                    f"Chat error: {str(e)}"
                )


# ==========================================
# 18. FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    🐰 Bunny PDF Chat · Powered by LangChain, Gemini & FAISS
    <br>
    Made with lots of love 💗
</div>
""", unsafe_allow_html=True)
