import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import tempfile

# Set up Streamlit page
st.title("PDF Q&A with Qwen 1B")
st.write("Upload a PDF file and ask questions about its content.")

# Initialize session state for storing vector store and chat history
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to process PDF and create vector store
def process_pdf(file):
    # Extract text
    text = extract_text_from_pdf(file)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store
    vector_store = Chroma.from_texts(chunks, embeddings)
    return vector_store

# Initialize Qwen 1B model
@st.cache_resource
def load_llm():
    model_path = "qwen-1_8b-chat-q4_0.gguf"  # Adjust to your GGUF file path
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure 'qwen-1_8b-chat-q4_0.gguf' is in the project directory.")
        return None
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        max_tokens=512,
        temperature=0.7,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True
    )
    return llm

# Process uploaded PDF
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Create vector store
        st.session_state.vector_store = process_pdf(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        st.success("PDF processed successfully!")

# Load LLM
llm = load_llm()

# Question input
question = st.text_input("Ask a question about the PDF content:")

# Process question and display answer
if question and st.session_state.vector_store is not None and llm is not None:
    with st.spinner("Generating answer..."):
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        # Get answer
        result = qa_chain({"query": question})
        answer = result["result"]
        
        # Store in chat history
        st.session_state.chat_history.append({"question": question, "answer": answer})
        
        # Display answer
        st.write("**Answer:**")
        st.write(answer)
        
        # Display source context (optional)
        st.write("**Source Context:**")
        for doc in result["source_documents"]:
            st.write(doc.page_content[:200] + "...")

# Display chat history
if st.session_state.chat_history:
    st.write("### Chat History")
    for i, chat in enumerate(st.session_state.chat_history):
        st.write(f"**Q{i+1}:** {chat['question']}")
        st.write(f"**A{i+1}:** {chat['answer']}")
        st.write("---")
