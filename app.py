import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Disable Streamlit's file watcher to avoid module path inspection error
os.environ["STREAMLIT_SERVER_WATCH_FILESYSTEM"] = "false"

# Set page configuration
st.set_page_config(page_title="PDF RAG App", page_icon="📄")

# Initialize session state for vector store and model
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Function to process PDF and create vector store
def process_pdf(pdf_file):
    # Extract text
    text = extract_text_from_pdf(pdf_file)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Function to load Qwen model and tokenizer
def load_model():
    if st.session_state.model is None:
        model_name = "Qwen/Qwen1.5-1.8B-Chat"
        st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_name)
        st.session_state.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32
        )
    return st.session_state.model, st.session_state.tokenizer

# Function to generate answer
def generate_answer(query, vector_store):
    model, tokenizer = load_model()
    
    # Retrieve relevant chunks
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = f"""Context: {context}

Question: {query}

Answer: """
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    # Decode response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace(prompt, "").strip()

# Streamlit UI
st.title("PDF RAG App with Qwen")
st.write("Upload a PDF file and ask questions based on its content.")

# PDF upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process PDF and create vector store
    with st.spinner("Processing PDF..."):
        st.session_state.vector_store = process_pdf("temp.pdf")
    
    # Remove temporary file
    os.remove("temp.pdf")
    
    st.success("PDF processed successfully!")

# Question input and answer generation
if st.session_state.vector_store:
    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, st.session_state.vector_store)
        st.write("**Answer:**")
        st.write(answer)
else:
    st.info("Please upload a PDF file to start asking questions.")
        st.write(answer)
else:
    st.info("Please upload a PDF file to start asking questions.")
