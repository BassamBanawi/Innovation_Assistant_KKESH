import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

# Load .env and validate API key
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "❌ OPENAI_API_KEY not set in .env file"

# Cache vectorstore loading
@st.cache_resource
def get_vectorstore_from_url(url):
    try:
        headers = {"User-Agent": os.getenv("USER_AGENT", "InvasionAssistantBot/1.0")}
        loader = WebBaseLoader(url, header_template=headers)
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        document_chunks = text_splitter.split_documents(document)

        vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
        return vector_store
    except Exception as e:
        st.error(f"Failed to load or process document from URL: {e}")
        return None

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

# Streamlit app config
st.set_page_config(page_title="Innovation Assistant 🤖", page_icon="🤖")
st.title("Innovation Assistant 🤖👨🏻‍💻")

# Website to scrape
website_url = "http://localhost:5173/"

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="👋 Hello! I'm here to help you explore and understand the ideas presented on this site. Feel free to ask anything!"),
        AIMessage(content="مرحبًا! هل لديك أي استفسارات أو مواضيع تحب مناقشتها حول الأفكار المطروحة في هذا الموقع؟ أنا هنا للمساعدة 😊"),
    ]

if "vector_store" not in st.session_state:
    with st.spinner("Loading and indexing website..."):
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

# User input
user_query = st.chat_input("Type your message here...")

if user_query:
    with st.spinner("Generating response..."):
        response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# Display conversation
for message in st.session_state.chat_history:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.write(message.content)