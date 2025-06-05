import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Streamlit page configuration
st.set_page_config(page_title="PDF Chatbot with Qwen 1B", layout="wide")
st.title("📄 PDF Chatbot with Qwen 1B")
st.markdown("Upload a PDF and ask questions about its content using a lightweight RAG system powered by Qwen 1B.")

# Initialize session state for storing vector store and chat history
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for PDF upload and processing
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                # Save the uploaded PDF to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Load and process the PDF
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()

                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_documents(documents)

                # Initialize embedding model (CPU-friendly)
                embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )

                # Create vector store with ChromaDB
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory=None  # In-memory for Streamlit Cloud
                )

                # Clean up temporary file
                os.unlink(tmp_file_path)
                st.success("PDF processed successfully!")

# Load Qwen 1B model and tokenizer with error handling
@st.cache_resource
def load_model():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize the model
llm = load_model()

# Check if model loading failed
if llm is None:
    st.error("Failed to load the language model. Please check your dependencies and try again.")
    st.stop()

# Define the prompt template for RAG
prompt_template = """Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Main chat interface
st.header("Ask Questions")
user_question = st.text_input("Enter your question about the PDF:")

if user_question and st.session_state.vector_store is not None:
    with st.spinner("Generating answer..."):
        # Set up the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        # Get the answer
        result = qa_chain({"query": user_question})
        answer = result["result"]
        sources = result["source_documents"]

        # Store in chat history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })

        # Display the answer
        st.markdown("**Answer:**")
        st.write(answer)

        # Display source documents
        with st.expander("Source Documents"):
            for i, doc in enumerate(sources):
                st.write(f"**Chunk {i+1}:** {doc.page_content[:300]}... (Page {doc.metadata['page']})")

# Display chat history
st.header("Chat History")
for i, chat in enumerate(st.session_state.chat_history):
    st.markdown(f"**Q{i+1}:** {chat['question']}")
    st.markdown(f"**A{i+1}:** {chat['answer']}")
    st.markdown("---")

# Instructions for use
st.markdown("""
### How to Use:
1. Upload a PDF file using the sidebar.
2. Click "Process PDF" to extract and index the content.
3. Enter your question in the text box and get answers based on the PDF content.
4. View the chat history and source documents for transparency.

**Note:** The system uses Qwen 1B (Qwen2.5-1.5B-Instruct) and runs entirely on CPU, making it suitable for Streamlit Cloud. If you encounter issues, ensure all dependencies are installed and consider running with `--server.fileWatcherType none`.
""")
